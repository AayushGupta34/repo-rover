"""Deterministic detector engine.

Runs the catalogue's `regex`, `absence`, `manifest` and `ast` detectors over
the confined repository before any agent starts. Every hit carries a file, a
line number and the verbatim text found there. The Inspector agent is then
asked only to judge these hits; it never searches for findings itself.

Hardening
- Patterns run on a shadow buffer: comments blanked, Unicode NFKC-normalised
  and lines capped at 2,000 characters. Line numbers are unchanged, and the
  evidence snippet is always taken from the real source line.
- Absence and manifest rules can be gated by repository archetype, so a rule
  about servers does not fire on a static page.
- Findings are ranked CRITICAL > HIGH > MEDIUM > LOW before the Inspector's
  context budget is applied, so a flood of low-value hits cannot push the
  serious ones out of the prompt.
- Secret values in snippets are masked before they reach a prompt or report.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repo_rover import complexity_scanner
from repo_rover.context_budget import DETECTOR_HITS_SHARE, budget_chars
from repo_rover.fingerprint import Fingerprint, fingerprint_repo, normalize_package
from repo_rover.read_ledger import RunContext
from repo_rover.repo_tools import MAX_LINE_CHARS, RepoPathError, iter_repo_files, load_text_file
from repo_rover.sanitize import extension_of, mask_secrets, shadow_text

DETERMINISTIC_TYPES = ("regex", "absence", "manifest", "ast")
MAX_FILES_INDEXED = 2000
MAX_TEXT_CHARS = 300_000
MAX_HITS_PER_DETECTOR = 5
MAX_HITS_PER_FINDING = 5
MAX_SNIPPET_LINES = 3
MAX_SNIPPET_LINE_CHARS = 240
ENGINE_TOOL_NAME = "DetectorEngine"
SEVERITIES = ("critical", "high", "medium", "low")


@dataclass(frozen=True)
class EngineOptions:
    """Switches for the engine's layers. All on by default; the benchmark
    turns them off to measure what each layer contributes."""

    strip_comments: bool = True
    normalize_unicode: bool = True
    archetype_gating: bool = True
    structural: bool = True
    manifestless_rules: bool = True


RAW_OPTIONS = EngineOptions(False, False, False, False, False)


class RepoIndex:
    """Lazily loaded text of every readable file in the confined repository."""

    def __init__(self, root: str | os.PathLike, options: EngineOptions = EngineOptions()):
        self.root = Path(os.path.realpath(root))
        self.options = options
        self.files: list[str] = []
        for relative in iter_repo_files(self.root):
            if len(self.files) >= MAX_FILES_INDEXED:
                break
            self.files.append(relative)
        self._cache: dict[str, tuple[str, list[str], str] | None] = {}
        self._shadow: dict[tuple[str, bool], str] = {}

    def load(self, relative: str) -> tuple[str, list[str], str] | None:
        """(sha256, lines, text) for a file, or None if it is not readable text."""
        if relative not in self._cache:
            try:
                sha256, lines = load_text_file(self.root / relative)
                self._cache[relative] = (sha256, lines, "\n".join(lines)[:MAX_TEXT_CHARS])
            except RepoPathError:
                self._cache[relative] = None
        return self._cache[relative]

    def scan_text(self, relative: str, scan_comments: bool = False) -> str | None:
        """The shadow buffer a pattern is matched against."""
        key = (relative, scan_comments)
        if key not in self._shadow:
            loaded = self.load(relative)
            if loaded is None:
                return None
            strip = self.options.strip_comments and not scan_comments
            self._shadow[key] = shadow_text(
                loaded[1], extension_of(relative), strip=strip, normalize=self.options.normalize_unicode
            )[:MAX_TEXT_CHARS]
        return self._shadow[key]

    def matching(self, globs: list[str] | None) -> list[str]:
        if not globs:
            return self.files
        lowered = [g.lower() for g in globs]
        return [
            relative
            for relative in self.files
            if any(fnmatch.fnmatch(relative.rsplit("/", 1)[-1].lower(), g) for g in lowered)
        ]


def _compile(pattern: Any) -> re.Pattern | None:
    try:
        return re.compile(str(pattern), re.MULTILINE)
    except re.error:
        return None


def _snippet(lines: list[str], start_line: int, end_line: int) -> str:
    parts = [lines[n - 1].strip()[:MAX_SNIPPET_LINE_CHARS] for n in range(start_line, end_line + 1) if n <= len(lines)]
    return mask_secrets("\n".join(part for part in parts if part))


def _location(index: RepoIndex, relative: str, text: str, match: re.Match) -> dict[str, Any]:
    _, lines, _ = index.load(relative)
    start_line = text.count("\n", 0, match.start()) + 1
    end_line = text.count("\n", 0, max(match.start(), match.end() - 1)) + 1
    end_line = min(end_line, start_line + MAX_SNIPPET_LINES - 1, len(lines))
    return {"file": relative, "line": start_line, "end_line": end_line, "snippet": _snippet(lines, start_line, end_line)}


def _run_regex(index: RepoIndex, detector: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    pattern = _compile(detector.get("pattern"))
    if pattern is None:
        return [], 0
    scan_comments = bool(detector.get("scan_comments"))
    hits: list[dict[str, Any]] = []
    total = 0
    for relative in index.matching(detector.get("file_globs")):
        text = index.scan_text(relative, scan_comments)
        if text is None:
            continue
        seen_lines: set[int] = set()
        for match in pattern.finditer(text):
            location = _location(index, relative, text, match)
            if location["line"] in seen_lines or not location["snippet"]:
                continue
            seen_lines.add(location["line"])
            total += 1
            if len(hits) < MAX_HITS_PER_DETECTOR:
                hits.append({"detector_type": "regex", **location, "detail": ""})
    return hits, total


def _first_match(index: RepoIndex, pattern: re.Pattern, files: list[str], scan_comments: bool = False):
    for relative in files:
        text = index.scan_text(relative, scan_comments)
        if text is None:
            continue
        match = pattern.search(text)
        if match:
            return _location(index, relative, text, match)
    return None


def _run_absence(index: RepoIndex, detector: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    pattern = _compile(detector.get("pattern"))
    if pattern is None:
        return [], 0
    files = index.matching(detector.get("file_globs"))
    scope = ", ".join(detector.get("file_globs") or ["all text files"])

    applies_at = None
    if detector.get("applies_if"):
        applies = _compile(detector["applies_if"])
        if applies is None:
            return [], 0
        applies_files = index.matching(detector["applies_if_globs"]) if detector.get("applies_if_globs") else files
        applies_at = _first_match(index, applies, applies_files)
        if applies_at is None:
            return [], 0  # the precondition is not met, so the rule does not apply

    if _first_match(index, pattern, files, bool(detector.get("scan_comments"))) is not None:
        return [], 0  # the expected safeguard is present

    detail = f"Expected pattern /{detector['pattern']}/ was found in none of the {len(files)} file(s) matching {scope}."
    if applies_at is not None:
        hit = {"detector_type": "absence", **applies_at, "detail": "Rule applies because of this line. " + detail}
    else:
        hit = {"detector_type": "absence", "file": None, "line": None, "end_line": None, "snippet": "", "detail": detail}
    return [hit], 1


def _package_key(name: Any) -> str:
    text = str(name)
    return text.lower() if text.startswith("@") else normalize_package(text)


def _run_manifest(index: RepoIndex, detector: dict[str, Any], fingerprint: Fingerprint):
    wanted = {str(name).lower() for name in detector.get("manifest_files") or []}
    present = {
        path: info
        for path, info in fingerprint.manifests.items()
        if info["kind"].lower() in wanted
        or (info["kind"].lower().startswith("requirements") and "requirements.txt" in wanted)
    }
    has_any = [_package_key(p) for p in detector.get("has_any") or []]
    lacks_all = [_package_key(p) for p in detector.get("lacks_all") or []]

    if not present:
        # A repository with source code and no manifest at all still lacks the
        # tooling a `lacks_all`-only rule looks for (tests, linter, ...).
        if index.options.manifestless_rules and not has_any and lacks_all and not fingerprint.manifests \
                and fingerprint.has_source_code:
            detail = "No dependency manifest exists, so none of these are declared: " + ", ".join(lacks_all) + "."
            return [{"detector_type": "manifest", "file": None, "line": None, "end_line": None, "snippet": "", "detail": detail}], 1
        return [], 0

    declared: dict[str, tuple[str, int]] = {}
    for path, info in sorted(present.items()):
        for package, line in info["packages"].items():
            declared.setdefault(_package_key(package), (path, line))

    found = [p for p in has_any if p in declared]
    if has_any and not found:
        return [], 0
    if any(p in declared for p in lacks_all):
        return [], 0

    path, line = declared[found[0]] if found else (sorted(present)[0], 1)
    loaded = index.load(path)
    snippet = ""
    if loaded is not None and loaded[1] and line <= len(loaded[1]):
        snippet = _snippet(loaded[1], line, line)
    parts = []
    if found:
        parts.append("Declares " + ", ".join(found) + ".")
    if lacks_all:
        parts.append("Declares none of: " + ", ".join(lacks_all) + ".")
    parts.append("Manifests checked: " + ", ".join(sorted(present)) + ".")
    hit = {
        "detector_type": "manifest",
        "file": path,
        "line": line if snippet else None,
        "end_line": line if snippet else None,
        "snippet": snippet,
        "detail": " ".join(parts),
    }
    return [hit], 1


def _run_ast(detector: dict[str, Any], structural: dict[str, list[dict[str, Any]]]):
    found = structural.get(str(detector.get("check")), [])
    hits = [
        {"detector_type": "ast", "file": h["file"], "line": h["line"], "end_line": h["end_line"],
         "snippet": mask_secrets(h["snippet"]), "detail": h["detail"]}
        for h in found[:MAX_HITS_PER_DETECTOR] if h.get("snippet")
    ]
    return hits, len(found)


def severity_of(flags: dict[str, Any] | None) -> str:
    """Severity from the catalogue's flags. Deterministic, never from a model."""
    flags = flags or {}
    security, loss = bool(flags.get("security_impact")), bool(flags.get("data_loss_risk"))
    sensitive = bool(flags.get("affects_payments_or_pii"))
    if security and (loss or sensitive):
        return "critical"
    if security or loss:
        return "high"
    if sensitive or flags.get("user_visible") or flags.get("concurrency_related"):
        return "medium"
    return "low"


def _score(entry: dict[str, Any], detector_types: set[str]) -> int:
    flags = entry.get("binary_flags") or {}
    score = 3 * bool(flags.get("security_impact")) + 3 * bool(flags.get("data_loss_risk"))
    score += 2 * bool(flags.get("affects_payments_or_pii")) + bool(flags.get("user_visible"))
    score += 2 if detector_types & {"regex", "ast"} else 1 if "manifest" in detector_types else 0
    return score


def run_detectors(
    root: str | os.PathLike,
    entries: list[dict[str, Any]],
    fingerprint: Fingerprint | None = None,
    ctx: RunContext | None = None,
    options: EngineOptions = EngineOptions(),
) -> list[dict[str, Any]]:
    """Run every deterministic detector and return one finding per catalogue
    entry that fired, most severe first, with ids D1..Dn.

    When `ctx` is given the findings are stored on the run and every snippet
    is recorded in the read ledger, so agents can quote hits as evidence.
    """
    index = RepoIndex(root, options)
    fingerprint = fingerprint or fingerprint_repo(root)
    structural = complexity_scanner.scan_repository(index) if options.structural else {}
    archetype = fingerprint.archetype
    findings: list[dict[str, Any]] = []

    for entry in entries:
        allowed = entry.get("archetypes") or []
        gated_out = options.archetype_gating and bool(allowed) and archetype not in allowed
        hits: list[dict[str, Any]] = []
        total = 0
        kinds: set[str] = set()
        for detector in entry.get("detection") or []:
            if not isinstance(detector, dict):
                continue
            kind = detector.get("type")
            if kind == "regex":
                found, count = _run_regex(index, detector)
            elif kind == "absence":
                found, count = ([], 0) if gated_out else _run_absence(index, detector)
            elif kind == "manifest":
                found, count = ([], 0) if gated_out else _run_manifest(index, detector, fingerprint)
            elif kind == "ast":
                found, count = _run_ast(detector, structural)
            else:
                continue
            if found:
                kinds.add(kind)
                hits.extend(found)
                total += count
        if not hits:
            continue
        findings.append(
            {
                "slug": entry.get("slug"),
                "title": entry.get("title"),
                "category": entry.get("category"),
                "vulnerability_type": entry.get("vulnerability_type"),
                "symptom": entry.get("symptom") or "",
                "catalog_id": entry.get("id"),
                "binary_flags": entry.get("binary_flags") or {},
                "fix_recommendation": entry.get("fix_recommendation") or "",
                "verification_steps": list(entry.get("verification_steps") or []),
                "detector_types": sorted(kinds),
                "hits": hits[:MAX_HITS_PER_FINDING],
                "total_hits": total,
                "severity": severity_of(entry.get("binary_flags")),
                "score": _score(entry, kinds),
            }
        )

    findings.sort(key=lambda f: (SEVERITIES.index(f["severity"]), -f["score"], str(f["slug"])))
    for number, finding in enumerate(findings, start=1):
        finding["id"] = f"D{number}"

    if ctx is not None:
        ctx.detector_findings = {finding["id"]: finding for finding in findings}
        ctx.artifacts["structural_targets"] = [
            {"check": check, "file": h["file"], "line": h["line"], "detail": h["detail"]}
            for check, hits in structural.items() for h in hits
        ][:40]
        for finding in findings:
            for hit in finding["hits"]:
                if not hit.get("file") or not hit.get("line"):
                    continue
                loaded = index.load(hit["file"])
                if loaded is None:
                    continue
                sha256, lines, _ = loaded
                served = {
                    n: mask_secrets(lines[n - 1][:MAX_LINE_CHARS])
                    for n in range(hit["line"], (hit["end_line"] or hit["line"]) + 1)
                    if n <= len(lines)
                }
                ctx.record_read(hit["file"], sha256, len(lines), served, ENGINE_TOOL_NAME, count_budget=False)
    return findings


def format_hit(hit: dict[str, Any]) -> str:
    if hit.get("file") and hit.get("line") and hit.get("snippet"):
        snippet = hit["snippet"].replace("\n", " \\n ")
        text = f"{hit['file']}:{hit['line']} | {snippet}"
        return text + (f"  ({hit['detail']})" if hit.get("detail") else "")
    return f"(no single location) {hit.get('detail', '')}"


def format_findings_for_judge(
    findings: list[dict[str, Any]], max_chars: int | None = None
) -> tuple[str, list[str]]:
    """Render findings as a compact block for the Inspector's prompt.

    Findings arrive most severe first, so when the context budget runs out it
    is the least severe that are left for the report's unjudged section.
    Returns (text, ids included).
    """
    limit = max_chars if max_chars is not None else budget_chars(DETECTOR_HITS_SHARE)
    blocks: list[str] = []
    included: list[str] = []
    used = 0
    for finding in findings:
        first = finding["hits"][0]
        more = finding["total_hits"] - 1
        block = (
            f"{finding['id']} | {finding['slug']} | {finding['title']} "
            f"| rule: {'+'.join(finding['detector_types'])} | catalogue severity: {finding.get('severity', 'medium')}\n"
            f"   {format_hit(first)[:420]}"
            + (f"\n   (+{more} more location(s))" if more > 0 else "")
        )
        if used + len(block) + 1 > limit and included:
            break
        blocks.append(block)
        included.append(finding["id"])
        used += len(block) + 1
    return "\n".join(blocks), included


def load_catalog_entries(prefer_db: bool = True) -> tuple[list[dict[str, Any]], str]:
    """The catalogue entries to run, and where they came from.

    Reads Postgres when it is reachable (so promoted entries and database ids
    are included) and falls back to the YAML seed file.
    """
    if prefer_db:
        try:
            from psycopg2.extras import RealDictCursor

            from repo_rover import db_tools

            with db_tools._connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT id, slug, title, category, vulnerability_type, stack_trigger, symptom,
                               binary_flags, detection, fix_recommendation, verification_steps, archetypes
                        FROM known_failure_nodes
                        WHERE slug IS NOT NULL AND jsonb_array_length(detection) > 0
                        ORDER BY id
                        """
                    )
                    rows = [dict(row) for row in cur.fetchall()]
            if rows:
                return rows, "database"
        except Exception as exc:
            print(f"Warning: catalogue not read from the database ({exc}); using the YAML seed.")

    from repo_rover import catalog

    data = catalog.load_catalog(catalog.catalog_path())
    return list(data.get("entries") or []), "yaml"
