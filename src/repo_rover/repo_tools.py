"""Repository reading tools, confined to the target directory and ledgered.

All four tools resolve paths through `resolve_repo_file` / `iter_repo_files`,
which refuse anything outside the run's target directory, anything inside
VCS/dependency/virtualenv/build directories, binaries, and secret files.
Every line a tool serves to the LLM is recorded in the run's read ledger and
counted against a per-run line budget.
"""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Iterator

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from repo_rover.context_budget import TOOL_OUTPUT_SHARE, budget_chars, fit_tool_output
from repo_rover.read_ledger import RunContext
from repo_rover.sanitize import mask_secrets
from repo_rover.runtime import (
    IGNORED_DIRECTORIES,
    IGNORED_SUFFIXES,
    MAX_LISTED_FILES,
    RepoDirectoryReadTool,
)

MAX_LINES_PER_READ = 200
MAX_CHARS_PER_READ = 16_000
MAX_LINE_CHARS = 400
MAX_FILE_BYTES = 2_000_000
MAX_SEARCH_MATCHES = 40
MAX_SEARCH_MATCHES_PER_FILE = 8
MAX_SEARCH_FILES = 2_000
MAX_OUTLINE_ENTRIES = 120
MAX_DIRECTORY_DEPTH = 12
MAX_SEARCH_QUERY_CHARS = 200
MAX_FILES_CONSIDERED_FOR_LISTING = 5000

# A quantified group that itself contains a quantifier, e.g. (a+)+ or (\w*)*:
# the classic shape of catastrophic backtracking.
_NESTED_QUANTIFIER = re.compile(r"\([^()]*[+*][^()]*\)\s*[+*{]")

_PRIORITY_NAMES = (
    "package.json", "pyproject.toml", "pipfile", "go.mod", "cargo.toml", "composer.json", "gemfile",
    "dockerfile", "docker-compose.yml", "docker-compose.yaml", "manage.py", "readme.md",
)
_PRIORITY_STEMS = ("main", "app", "index", "server", "wsgi", "asgi", "settings", "routes", "schema")
_PRIORITY_DIRECTORIES = ("src", "app", "server", "api", "lib", "pages", "routes", "backend", "frontend")


def listing_priority(relative: str) -> tuple[int, int, str]:
    """Order for the directory listing: manifests and entry points first, then
    architectural directories, then everything else. A flood of decoy files
    with early alphabetical names cannot push the real code out of the listing."""
    parts = relative.lower().split("/")
    name = parts[-1]
    stem = name.rsplit(".", 1)[0]
    if name in _PRIORITY_NAMES or name.startswith("requirements") or (len(parts) <= 2 and stem in _PRIORITY_STEMS):
        rank = 0
    elif parts[0] in _PRIORITY_DIRECTORIES or len(parts) == 1:
        rank = 1
    else:
        rank = 2
    return rank, len(parts), relative.lower()

SECRET_FILE_PATTERNS = (
    ".env", ".env.*", "*.pem", "*.key", "*.pfx", "*.p12", "*.keystore",
    "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*", ".npmrc", ".pypirc",
    ".netrc", "secrets.*",
)
SECRET_FILE_ALLOWED = (".env.example", ".env.sample", ".env.template")

_OUTLINE_PATTERN = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:public\s+|private\s+|protected\s+|static\s+)*"
    r"(?:async\s+)?"
    r"(?:function\b|class\b|def\b|func\b|fn\b|interface\b|struct\b|enum\b|impl\b|trait\b"
    r"|(?:const|let|var)\s+[\w$]+\s*=\s*(?:async\s*)?(?:\(|function\b|[\w$]+\s*=>))"
)


class RepoPathError(ValueError):
    """A requested path is not readable under the confinement rules."""


def _is_secret_file(name: str) -> bool:
    lowered = name.lower()
    if lowered in SECRET_FILE_ALLOWED:
        return False
    return any(fnmatch.fnmatch(lowered, pattern) for pattern in SECRET_FILE_PATTERNS)


def _is_skipped_file(name: str) -> bool:
    return name.lower().endswith(IGNORED_SUFFIXES) or _is_secret_file(name)


def resolve_repo_file(root: Path, file_path: Any) -> tuple[Path, str]:
    """Resolve an LLM-supplied path to (absolute path, repo-relative posix path).

    Raises RepoPathError unless the fully resolved path (symlinks and `..`
    followed) is a regular, readable-by-policy file inside `root`.
    """
    raw = str(file_path or "").strip().strip("`'\"")
    if not raw:
        raise RepoPathError("file_path is required.")
    try:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = Path(os.path.realpath(candidate))
        relative = resolved.relative_to(root)
    except (ValueError, OSError):
        raise RepoPathError(
            f"'{raw}' is outside the repository being analyzed and cannot be read."
        ) from None

    parts = relative.parts
    if not parts:
        raise RepoPathError(f"'{raw}' is a directory, not a file.")
    blocked = next((p for p in parts[:-1] if p in IGNORED_DIRECTORIES), None)
    if blocked:
        raise RepoPathError(f"'{raw}' is inside the excluded directory '{blocked}'.")
    if not resolved.is_file():
        raise RepoPathError(f"'{raw}' is not a file in the repository.")
    if _is_secret_file(parts[-1]):
        raise RepoPathError(f"'{raw}' looks like a secrets file and is never read.")
    if parts[-1].lower().endswith(IGNORED_SUFFIXES):
        raise RepoPathError(f"'{raw}' is a binary, lock or generated file.")
    return resolved, relative.as_posix()


def iter_repo_files(root: Path) -> Iterator[str]:
    """Yield repo-relative posix paths of readable files, in a stable order."""
    root = Path(os.path.realpath(root))  # e.g. Windows 8.3 short temp paths
    seen_directories = {os.path.normcase(str(root))}
    for current, dirs, files in os.walk(root):
        depth = len(Path(current).relative_to(root).parts)
        kept = []
        for directory in sorted(dirs):
            if directory in IGNORED_DIRECTORIES or depth + 1 > MAX_DIRECTORY_DEPTH:
                continue
            real = os.path.realpath(os.path.join(current, directory))
            key = os.path.normcase(real)
            try:
                Path(real).relative_to(root)  # a link or junction may point outside the repository
            except ValueError:
                continue
            if key in seen_directories:
                continue  # a symlink or Windows junction loop
            seen_directories.add(key)
            kept.append(directory)
        dirs[:] = kept
        for filename in sorted(files):
            if _is_skipped_file(filename):
                continue
            full = Path(current) / filename
            try:
                # A symlink may point outside the repository.
                Path(os.path.realpath(full)).relative_to(root)
            except (ValueError, OSError):
                continue
            yield full.relative_to(root).as_posix()


def load_text_file(path: Path) -> tuple[str, list[str]]:
    """Return (sha256, lines) for a text file, enforcing the size cap."""
    try:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise RepoPathError(
                f"file is {size} bytes, over the {MAX_FILE_BYTES} byte read limit."
            )
        data = path.read_bytes()
    except OSError as exc:
        raise RepoPathError(f"file could not be read: {exc}") from None
    if b"\x00" in data[:8192]:
        raise RepoPathError("file is binary.")
    lines = data.decode("utf-8", errors="replace").splitlines()
    return hashlib.sha256(data).hexdigest(), lines


def _clip(line: str) -> str:
    return line[:MAX_LINE_CHARS]


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class _LedgeredTool(BaseTool):
    _ctx: RunContext = PrivateAttr()

    def __init__(self, ctx: RunContext, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._ctx = ctx


class LedgeredDirectoryReadTool(RepoDirectoryReadTool):
    """RepoDirectoryReadTool pinned to the run's target directory.

    Any `directory` argument is ignored, secret files and escaping symlinks
    are left out, and the listing is recorded in the read ledger.
    """

    _ctx: RunContext = PrivateAttr()

    def __init__(self, ctx: RunContext, **kwargs: Any) -> None:
        super().__init__(directory=str(ctx.root), **kwargs)
        self._ctx = ctx

    def _run(self, **kwargs: Any) -> Any:
        root = self._ctx.root
        listed: list[str] = []
        truncated = False
        candidates: list[str] = []
        for relative in iter_repo_files(root):
            candidates.append(relative)
            if len(candidates) >= MAX_FILES_CONSIDERED_FOR_LISTING:
                break
        candidates.sort(key=listing_priority)
        truncated = len(candidates) > MAX_LISTED_FILES
        listed = candidates[:MAX_LISTED_FILES]
        self._ctx.record_listing(listed)
        if not listed:
            return f"No readable source files found under {root}."
        note = f"\n(Listing truncated at {MAX_LISTED_FILES} files.)" if truncated else ""
        return fit_tool_output(
            f"Files under {root} (paths are relative to it; pass them unchanged "
            "as file_path to the other repository tools):\n- "
            + "\n- ".join(listed)
            + note
        )


class ReadRepoFileInput(BaseModel):
    file_path: str = Field(..., description="Repo-relative path from the directory listing.")
    start_line: int = Field(1, description="1-based line to start reading from.")
    line_count: int = Field(
        MAX_LINES_PER_READ,
        description=f"Number of lines to read; at most {MAX_LINES_PER_READ} per call.",
    )


class ReadRepoFileTool(_LedgeredTool):
    name: str = "ReadRepoFileTool"
    description: str = (
        "Read a window of lines from one file in the repository being analyzed. "
        f"Returns at most {MAX_LINES_PER_READ} numbered lines per call; page "
        "through longer files with start_line. Only files read with this tool "
        "may be reported as reviewed, and quotes must be copied from its output."
    )
    args_schema: type[BaseModel] = ReadRepoFileInput

    def _run(self, file_path: str, start_line: Any = 1, line_count: Any = None) -> str:
        ctx = self._ctx
        try:
            resolved, relative = resolve_repo_file(ctx.root, file_path)
            sha256, lines = load_text_file(resolved)
        except RepoPathError as exc:
            return f"Error: {exc}"

        remaining = ctx.budget_remaining()
        if remaining <= 0:
            return (
                "Error: this run's read budget is exhausted. Do not request more "
                "file content; finish with what you have already read."
            )

        total = len(lines)
        start = max(1, _as_int(start_line, 1))
        count = max(1, _as_int(line_count, MAX_LINES_PER_READ))
        count = min(count, MAX_LINES_PER_READ, remaining)
        if total == 0:
            return f"{relative} is empty (0 lines)."
        if start > total:
            return f"Error: {relative} has only {total} lines; start_line {start} is past the end."

        served: dict[int, str] = {}
        rendered: list[str] = []
        used_chars = 0
        for number in range(start, min(total, start + count - 1) + 1):
            text = mask_secrets(_clip(lines[number - 1]))
            used_chars += len(text) + 8
            if served and used_chars > min(MAX_CHARS_PER_READ, budget_chars(TOOL_OUTPUT_SHARE)):
                break
            served[number] = text
            rendered.append(f"{number:>5}| {text}")

        end = max(served)
        ctx.record_read(relative, sha256, total, served, self.name)
        header = (
            f"{relative} - lines {start}-{end} of {total} "
            f"(sha256 {sha256[:12]}). Line numbers before '|' are not file content."
        )
        footer = (
            f"\n[{total - end} more lines. Continue with start_line={end + 1}.]"
            if end < total
            else "\n[End of file.]"
        )
        return header + "\n" + "\n".join(rendered) + footer


class SearchRepoInput(BaseModel):
    query: str = Field(..., description="Text to search for (case-insensitive).")
    is_regex: bool = Field(False, description="Treat query as a regular expression.")
    path_glob: str = Field(
        "", description="Optional glob on the repo-relative path, e.g. 'src/*.py'."
    )


class SearchRepoTool(_LedgeredTool):
    name: str = "SearchRepoTool"
    description: str = (
        "Search the text files of the repository being analyzed for a string or "
        f"regular expression. Returns up to {MAX_SEARCH_MATCHES} matching lines as "
        "'path:line| text'. Use it to locate code before reading it."
    )
    args_schema: type[BaseModel] = SearchRepoInput

    def _run(self, query: str, is_regex: Any = False, path_glob: str = "") -> str:
        ctx = self._ctx
        query = str(query or "")
        if not query.strip():
            return "Error: query is required."
        as_regex = is_regex is True or str(is_regex).lower() == "true"
        if len(query) > MAX_SEARCH_QUERY_CHARS:
            return f"Error: query is longer than {MAX_SEARCH_QUERY_CHARS} characters."
        if as_regex and _NESTED_QUANTIFIER.search(query):
            return "Error: regular expressions with nested quantifiers such as (a+)+ are not allowed."
        try:
            pattern = re.compile(query if as_regex else re.escape(query), re.IGNORECASE)
        except (re.error, RecursionError, OverflowError) as exc:
            return f"Error: invalid regular expression: {exc}"
        if ctx.budget_remaining() <= 0:
            return "Error: this run's read budget is exhausted."

        glob = str(path_glob or "").strip().replace("\\", "/")
        results: list[str] = []
        scanned = 0
        capped = False
        for relative in iter_repo_files(ctx.root):
            if glob and not fnmatch.fnmatch(relative, glob):
                continue
            scanned += 1
            if scanned > MAX_SEARCH_FILES:
                capped = True
                break
            try:
                sha256, lines = load_text_file(ctx.root / relative)
            except RepoPathError:
                continue
            hits: dict[int, str] = {}
            for number, line in enumerate(lines, start=1):
                text = _clip(line)
                if pattern.search(text):
                    hits[number] = mask_secrets(text)
                    if len(hits) >= MAX_SEARCH_MATCHES_PER_FILE:
                        break
            if not hits:
                continue
            allowed = min(MAX_SEARCH_MATCHES - len(results), ctx.budget_remaining())
            hits = dict(list(hits.items())[:allowed])
            if hits:
                ctx.record_read(relative, sha256, len(lines), hits, self.name)
                results.extend(f"{relative}:{n}| {t}" for n, t in hits.items())
            if len(results) >= MAX_SEARCH_MATCHES or ctx.budget_remaining() <= 0:
                capped = True
                break

        if not results:
            return f"No matches for '{query}'."
        note = "\n[Results capped; narrow the query or use path_glob.]" if capped else ""
        return fit_tool_output(f"{len(results)} matching lines:\n" + "\n".join(results) + note)


class OutlineFileInput(BaseModel):
    file_path: str = Field(..., description="Repo-relative path from the directory listing.")


class OutlineFileTool(_LedgeredTool):
    name: str = "OutlineFileTool"
    description: str = (
        "Show the structure of one file in the repository being analyzed: its "
        "total line count and the line numbers of its classes and functions. Use "
        "it to decide which line ranges are worth reading."
    )
    args_schema: type[BaseModel] = OutlineFileInput

    def _run(self, file_path: str) -> str:
        ctx = self._ctx
        try:
            resolved, relative = resolve_repo_file(ctx.root, file_path)
            sha256, lines = load_text_file(resolved)
        except RepoPathError as exc:
            return f"Error: {exc}"
        if ctx.budget_remaining() <= 0:
            return "Error: this run's read budget is exhausted."

        numbers = self._python_outline(lines) if relative.endswith(".py") else None
        if numbers is None:
            numbers = [
                n for n, line in enumerate(lines, start=1) if _OUTLINE_PATTERN.match(line)
            ]
        limit = min(MAX_OUTLINE_ENTRIES, ctx.budget_remaining())
        truncated = len(numbers) > limit
        served = {n: mask_secrets(_clip(lines[n - 1])) for n in numbers[:limit]}
        ctx.record_read(relative, sha256, len(lines), served, self.name)

        header = f"{relative} - {len(lines)} lines (sha256 {sha256[:12]})."
        if not served:
            return header + " No class or function definitions detected."
        body = "\n".join(f"{n:>5}| {t.strip()}" for n, t in served.items())
        note = "\n[Outline truncated.]" if truncated else ""
        return fit_tool_output(header + " Definitions:\n" + body + note)

    @staticmethod
    def _python_outline(lines: list[str]) -> list[int] | None:
        try:
            tree = ast.parse("\n".join(lines))
        except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
            return None
        kinds = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        return sorted(
            {node.lineno for node in ast.walk(tree) if isinstance(node, kinds)}
        )


def build_repo_tools(ctx: RunContext) -> dict[str, BaseTool]:
    """The confined, ledgered tool set for one run."""
    return {
        "list": LedgeredDirectoryReadTool(ctx),
        "read": ReadRepoFileTool(ctx),
        "search": SearchRepoTool(ctx),
        "outline": OutlineFileTool(ctx),
    }
