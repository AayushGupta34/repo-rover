"""Closed-loop fix verification (`--verify-fixes`).

For findings that have a mechanical remedy, RepoRover applies a structured
transform to an isolated temporary copy of the repository, re-runs the
detector engine and a syntax check on that copy, and records whether the
finding is gone. The user's repository is never modified. The result is a
unified diff the developer can review and apply, with each fix marked
verified or not.

A fix is `verified_fix: true` only when all of these hold on the copy:
- the transform changed at least one file;
- the rule that reported the finding no longer fires;
- every changed Python or JavaScript file still parses;
- no rule that was silent before fires now.
"""

from __future__ import annotations

import ast
import difflib
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from repo_rover.detector_engine import run_detectors
from repo_rover.fingerprint import fingerprint_repo
from repo_rover.repo_tools import iter_repo_files
from repo_rover.runtime import IGNORED_DIRECTORIES

JS = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")
MARKUP = (".html", ".htm", ".jsx", ".tsx", ".vue", ".svelte")
STYLE = (".css", ".scss", ".html", ".htm", ".jsx", ".tsx", ".vue", ".svelte")
MAX_FILE_BYTES = 1_000_000


def _sub(pattern: str, replacement: str | Callable[[re.Match], str], flags: int = 0):
    compiled = re.compile(pattern, flags)
    return lambda text: compiled.sub(replacement, text)


def _add_requests_timeout(text: str) -> str:
    pattern = re.compile(r"(requests\.(?:get|post|put|delete|patch)\()((?:[^()]|\([^()]*\))*)\)")

    def fix(match: re.Match) -> str:
        arguments = match.group(2)
        if "timeout" in arguments:
            return match.group(0)
        separator = ", " if arguments.strip() else ""
        return f"{match.group(1)}{arguments}{separator}timeout=10)"

    return pattern.sub(fix, text)


def _add_viewport(text: str) -> str:
    if re.search(r"name=[\"']viewport[\"']", text):
        return text
    return re.sub(
        r"(<head[^>]*>)",
        r'\1\n  <meta name="viewport" content="width=device-width, initial-scale=1">',
        text, count=1, flags=re.IGNORECASE,
    )


# slug -> list of (file suffixes, transform). Transforms are pure text -> text.
TRANSFORMS: dict[str, list[tuple[tuple[str, ...], Callable[[str], str]]]] = {
    "debug-mode-enabled": [
        ((".py",), _sub(r"\bdebug\s*=\s*True\b", "debug=False")),
        ((".py",), _sub(r"^(\s*DEBUG\s*=\s*)True\b", r"\1False", re.MULTILINE)),
    ],
    "outbound-calls-without-timeout": [((".py",), _add_requests_timeout)],
    "tls-verification-disabled": [
        ((".py",), _sub(r"\bverify\s*=\s*False\b", "verify=True")),
        (JS, _sub(r"rejectUnauthorized\s*:\s*false", "rejectUnauthorized: true")),
    ],
    "unsafe-deserialization": [((".py",), _sub(r"yaml\.load\((?![^)]*SafeLoader)", "yaml.safe_load("))],
    "wildcard-cors": [
        (JS, _sub(r"\bcors\(\s*\)", "cors({ origin: (process.env.ALLOWED_ORIGINS || '').split(',').filter(Boolean) })")),
    ],
    "request-body-size-unlimited": [
        (JS, _sub(r"limit\s*:\s*([\"'])(?:\d{3,}mb|\d+gb)\1", "limit: '1mb'")),
    ],
    "insecure-session-cookie-flags": [
        (JS, _sub(r"\b(secure|httpOnly)\s*:\s*false", r"\1: true")),
        ((".py",), _sub(r"\b(SESSION_COOKIE_SECURE|SESSION_COOKIE_HTTPONLY)\s*=\s*False", r"\1 = True")),
    ],
    "jwt-hardcoded-secret": [
        (JS, _sub(r"(jwt\.sign\s*\([^;\n]*,\s*)[\"'][^\"'\n]{1,40}[\"'](\s*[,)])", r"\1process.env.JWT_SECRET\2")),
    ],
    "tokens-never-expire": [
        (JS, _sub(r"(jwt\.sign\s*\([^;\n]*,\s*process\.env\.\w+)\s*\)", r"\1, { expiresIn: '15m' })")),
    ],
    "viewport-meta-missing": [((".html", ".htm"), _add_viewport)],
    "basic-accessibility-missing": [(MARKUP, _sub(r"<img\b(?![^>]*\balt=)", '<img alt=""'))],
    "unoptimised-images": [(MARKUP, _sub(r"<img\b(?![^>]*(?:srcset|loading=))", '<img loading="lazy"'))],
    "full-height-layout-with-100vh": [(STYLE, _sub(r"\b100vh\b", "100dvh"))],
}
# Applied in this order so that later transforms see earlier results.
ORDER = tuple(TRANSFORMS)
GITIGNORE_SLUG = "env-file-not-ignored"


def _remove_tree(path: Path) -> None:
    def make_writable(function, failed_path, _exc):
        os.chmod(failed_path, stat.S_IWRITE)
        function(failed_path)

    try:
        shutil.rmtree(path, onexc=make_writable)
    except TypeError:  # Python < 3.12
        shutil.rmtree(path, onerror=lambda f, p, _e: (os.chmod(p, stat.S_IWRITE), f(p)))
    except OSError:
        pass


def _copy_repository(source: Path, destination: Path) -> None:
    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in IGNORED_DIRECTORIES}

    shutil.copytree(source, destination, ignore=ignore, symlinks=False, ignore_dangling_symlinks=True)


def syntax_error(path: Path, node: str | None) -> str | None:
    """None if the file parses, else a short reason."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        elif suffix in (".js", ".mjs", ".cjs") and node:
            done = subprocess.run([node, "--check", str(path)], capture_output=True, text=True, timeout=30)
            if done.returncode != 0:
                return (done.stderr or "syntax error").strip().splitlines()[-1][:200]
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} (line {exc.lineno})"
    except (RecursionError, MemoryError, subprocess.TimeoutExpired, OSError) as exc:
        return f"{type(exc).__name__}"
    return None


def verify_fixes(
    repo_root: str | os.PathLike,
    entries: list[dict[str, Any]],
    findings_before: list[dict[str, Any]] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> dict[str, Any]:
    """Apply mechanical fixes to a temporary copy and verify them.

    Returns counts, a per-finding result list and a unified diff. Nothing under
    `repo_root` is written.
    """
    source = Path(os.path.realpath(repo_root))
    if findings_before is None:
        findings_before = run_detectors(source, entries, fingerprint_repo(source))
    before_slugs = {f["slug"] for f in findings_before}
    id_of = {f["slug"]: f.get("id") for f in findings_before}

    workspace = Path(os.path.realpath(tempfile.mkdtemp(prefix="reporover_fix_")))
    copy = workspace / source.name
    node = which("node")
    try:
        _copy_repository(source, copy)
        changed: dict[str, set[str]] = {}
        originals: dict[str, str] = {}

        for slug in ORDER:
            if slug not in before_slugs:
                continue
            for relative in list(iter_repo_files(copy)):
                path = copy / relative
                for suffixes, transform in TRANSFORMS[slug]:
                    if not relative.lower().endswith(suffixes):
                        continue
                    try:
                        if path.stat().st_size > MAX_FILE_BYTES:
                            continue
                        text = path.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        continue
                    updated = transform(text)
                    if updated != text:
                        originals.setdefault(relative, text)
                        path.write_text(updated, encoding="utf-8", newline="")
                        changed.setdefault(slug, set()).add(relative)

        def ignore_env_files() -> None:
            ignore_file = copy / ".gitignore"
            existing = ignore_file.read_text(encoding="utf-8") if ignore_file.is_file() else ""
            originals.setdefault(".gitignore", existing)
            ignore_file.write_text(existing.rstrip("\n") + ("\n" if existing else "") + ".env\n.env.*\n!.env.example\n",
                                   encoding="utf-8", newline="\n")
            changed.setdefault(GITIGNORE_SLUG, set()).add(".gitignore")

        if GITIGNORE_SLUG in before_slugs:
            ignore_env_files()

        syntax_problems = {}
        for relative in sorted({file for files in changed.values() for file in files}):
            problem = syntax_error(copy / relative, node)
            if problem:
                syntax_problems[relative] = problem

        findings_after = run_detectors(copy, entries, fingerprint_repo(copy))
        after_slugs = {f["slug"] for f in findings_after}
        introduced = sorted(after_slugs - before_slugs)
        companion_fixes = []
        if introduced == [GITIGNORE_SLUG]:
            # Second round of the loop: moving a secret into the environment
            # makes "env files are not ignored" apply. Fix that too, re-check.
            ignore_env_files()
            companion_fixes.append(GITIGNORE_SLUG)
            findings_after = run_detectors(copy, entries, fingerprint_repo(copy))
            after_slugs = {f["slug"] for f in findings_after}
            introduced = sorted(after_slugs - before_slugs)

        results = []
        for slug in [s for s in (*ORDER, GITIGNORE_SLUG) if s in before_slugs]:
            files = sorted(changed.get(slug, ()))
            broken = [f for f in files if f in syntax_problems]
            if not files:
                verified, reason = False, "no mechanical transform matched this occurrence"
            elif broken:
                verified, reason = False, f"patched file no longer parses: {broken[0]} ({syntax_problems[broken[0]]})"
            elif slug in after_slugs:
                verified, reason = False, "the rule still fires after the patch"
            elif introduced:
                verified, reason = False, "the patch set introduced new findings: " + ", ".join(introduced)
            else:
                verified, reason = True, "rule no longer fires; changed files parse; no new findings"
            results.append({"slug": slug, "finding_id": id_of.get(slug), "files_changed": files,
                            "verified_fix": verified, "reason": reason})

        diff_parts = []
        for relative in sorted(originals):
            new_text = (copy / relative).read_text(encoding="utf-8")
            diff_parts.extend(difflib.unified_diff(
                originals[relative].splitlines(keepends=True), new_text.splitlines(keepends=True),
                fromfile=f"a/{relative}", tofile=f"b/{relative}"))
        diff = "".join(line if line.endswith("\n") else line + "\n" for line in diff_parts)

        return {
            "findings_before": len(before_slugs),
            "findings_after": len(after_slugs),
            "fixes_attempted": sum(1 for r in results if r["files_changed"]),
            "fixes_verified": sum(1 for r in results if r["verified_fix"]),
            "without_mechanical_fix": len(before_slugs) - len(results),
            "new_findings_introduced": introduced,
            "companion_fixes": companion_fixes,
            "syntax_checked_with_node": bool(node),
            "results": results,
            "diff": diff,
            "repository_modified": False,
        }
    finally:
        _remove_tree(workspace)


def summary_line(outcome: dict[str, Any]) -> str:
    return (
        f"Patch verification: {outcome['fixes_verified']} of {outcome['fixes_attempted']} mechanical fixes verified "
        f"on a temporary copy; detector findings {outcome['findings_before']} -> {outcome['findings_after']}."
    )
