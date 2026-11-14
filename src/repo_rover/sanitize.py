"""Sanitising helpers shared by the tools, the detector engine and the report.

- strip_comments / code_only: blank out comments (and optionally string
  contents) while keeping every character position and line number.
- shadow_text: the buffer detectors scan: long lines capped, Unicode
  NFKC-normalised, comments blanked. Line numbers match the original file.
- mask_secrets: replace literal secret values with [REDACTED_SECRET] while
  keeping the variable name, so agents can see that a credential is exposed
  without seeing it.
- sanitize_text: make model-written prose safe to place in a prompt a
  developer will paste elsewhere. It never rejects text; it rewrites in place.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Any

MAX_REGEX_LINE_CHARS = 2000
REDACTED = "[REDACTED_SECRET]"
SANITIZED = "[SANITIZED]"
LINK_REMOVED = "[link removed]"

HASH_COMMENT_EXTENSIONS = {".py", ".rb", ".sh", ".yml", ".yaml", ".toml", ".txt", ".cfg", ".ini", ".tf", ".dockerfile"}
SLASH_COMMENT_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".java", ".go", ".php", ".c", ".h",
                            ".cpp", ".cs", ".swift", ".kt", ".rs", ".dart", ".scss"}
BLOCK_ONLY_EXTENSIONS = {".css"}
HTML_EXTENSIONS = {".html", ".htm", ".vue", ".svelte", ".xml"}
SQL_EXTENSIONS = {".sql"}

ALLOWED_LINK_HOSTS = ("cwe.mitre.org", "owasp.org", "localhost", "127.0.0.1")

_INJECTION_PATTERNS = (
    re.compile(r"(?i)\bignore\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)"),
    re.compile(r"(?i)\bdisregard\s+(all\s+)?(the\s+)?(previous|prior|above|system)\s+(instructions|prompts?|rules)"),
    re.compile(r"(?i)\bsystem\s+override\b"),
    re.compile(r"(?i)\bbefore\s+fixing\s+this\b"),
    re.compile(r"(?i)\bnew\s+instructions\s*:"),
    re.compile(r"(?i)\byou\s+are\s+now\s+(in\s+)?(developer|dan|unrestricted|jailbreak)\b"),
)
_SHELL_PATTERNS = (
    re.compile(r"(?i)\b(curl|wget)\b[^|;&\n]{0,200}\|\s*(sudo\s+)?(ba|z)?sh\b"),
    re.compile(r"(?i)\b(curl|wget)\b[^\n]{0,200}&&[^\n]{0,120}"),
    re.compile(r"(?i)\brm\s+-rf?\s+[/~][^\s]*"),
)
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_URL = re.compile(r"https?://[^\s)>\]\"'`]+")

_PROVIDER_KEYS = re.compile(
    r"(AIza[0-9A-Za-z_\-]{30,}|\bgsk_[0-9A-Za-z]{20,}|\bsk-[0-9A-Za-z_\-]{20,}|\bsk_live_[0-9A-Za-z]{10,}"
    r"|\bghp_[0-9A-Za-z]{30,}|\bAKIA[0-9A-Z]{16}\b|\bxox[baprs]-[0-9A-Za-z-]{10,})"
)
_URL_PASSWORD = re.compile(r"((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqps?)://[^:\s/@]+:)([^@\s]+)(@)")
_ASSIGNED_SECRET = re.compile(
    r"(?i)(\b[\w.\-]*(?:api_?key|secret|passw(?:or)?d|token|private_?key)[\w\-]*[\"'\]]*\s*[:=]\s*)([\"'])([^\"'\n]{8,})\2"
)
_SECRET_ENV_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "DATABASE_URL")


def extension_of(name: str) -> str:
    lowered = name.lower().rsplit("/", 1)[-1]
    if lowered == "dockerfile" or lowered.startswith("dockerfile."):
        return ".dockerfile"
    return "." + lowered.rsplit(".", 1)[-1] if "." in lowered else ""


def _blank(chars: list[str], start: int, end: int) -> None:
    for i in range(start, end):
        if chars[i] != "\n":
            chars[i] = " "


def strip_comments(text: str, extension: str, blank_strings: bool = False) -> str:
    """Replace comments with spaces, preserving length and newlines.

    String literals are respected, so `"http://x"` and `'#fff'` survive. With
    `blank_strings` the contents of string literals are blanked too, leaving a
    code-only buffer for structural scanning.
    """
    ext = extension.lower()
    if ext in HTML_EXTENSIONS:
        chars = list(text)
        for match in re.finditer(r"<!--.*?-->", text, re.DOTALL):
            _blank(chars, match.start(), match.end())
        return "".join(chars)

    hash_comments = ext in HASH_COMMENT_EXTENSIONS
    slash_comments = ext in SLASH_COMMENT_EXTENSIONS
    block_comments = slash_comments or ext in BLOCK_ONLY_EXTENSIONS or ext in SQL_EXTENSIONS
    dash_comments = ext in SQL_EXTENSIONS
    if not (hash_comments or block_comments or dash_comments):
        return text

    chars = list(text)
    n = len(text)
    i = 0
    quotes = "'\"`" if slash_comments else "'\""
    while i < n:
        c = text[i]
        two = text[i:i + 2]
        if c in quotes:
            triple = text[i:i + 3] if ext == ".py" and text[i:i + 3] in ('"""', "'''") else None
            close = triple or c
            j = i + len(close)
            while j < n:
                if text[j] == "\\" and not triple:
                    j += 2
                    continue
                if text.startswith(close, j):
                    break
                if text[j] == "\n" and not triple and c != "`":
                    break  # unterminated single-line string
                j += 1
            end = min(n, j + len(close))
            if blank_strings:
                _blank(chars, i + len(close), min(n, j))
            i = end
            continue
        if hash_comments and c == "#":
            j = text.find("\n", i)
            j = n if j == -1 else j
            _blank(chars, i, j)
            i = j
            continue
        if (slash_comments and two == "//") or (dash_comments and two == "--"):
            j = text.find("\n", i)
            j = n if j == -1 else j
            _blank(chars, i, j)
            i = j
            continue
        if block_comments and two == "/*":
            j = text.find("*/", i + 2)
            j = n if j == -1 else j + 2
            _blank(chars, i, j)
            i = j
            continue
        i += 1
    return "".join(chars)


def shadow_text(lines: list[str], extension: str, strip: bool = True, normalize: bool = True) -> str:
    """The text detectors scan. Same number of lines as the source.

    - each line is capped at MAX_REGEX_LINE_CHARS, which bounds regex work;
    - each line is NFKC-normalised, so look-alike Unicode identifiers such as
      a script-e `eval` are matched by ASCII patterns;
    - comments are blanked, so commented-out code is not reported.
    """
    prepared = []
    for line in lines:
        line = line[:MAX_REGEX_LINE_CHARS]
        if normalize and not line.isascii():
            line = unicodedata.normalize("NFKC", line).replace("\n", " ").replace("\r", " ")
        prepared.append(line)
    text = "\n".join(prepared)
    return strip_comments(text, extension) if strip else text


def _env_secret_values() -> list[str]:
    return sorted(
        {value for name, value in os.environ.items()
         if any(hint in name.upper() for hint in _SECRET_ENV_HINTS) and len(value) >= 8},
        key=len, reverse=True,
    )


def mask_secrets(text: Any) -> str:
    """Mask literal secret values; keep names, structure and line count."""
    text = str(text if text is not None else "")
    if not text:
        return text
    for value in _env_secret_values():
        if value in text:
            text = text.replace(value, REDACTED)
    text = _URL_PASSWORD.sub(lambda m: m.group(1) + REDACTED + m.group(3), text)
    text = _PROVIDER_KEYS.sub(REDACTED, text)
    return _ASSIGNED_SECRET.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}{m.group(2)}", text)


def _vet_url(match: re.Match) -> str:
    url = match.group(0)
    host = re.sub(r"^https?://", "", url).split("/", 1)[0].split(":", 1)[0].lower()
    if host in ALLOWED_LINK_HOSTS or host.endswith(".owasp.org"):
        return url
    return LINK_REMOVED


def sanitize_text(text: Any, max_length: int = 400) -> str:
    """Make model-written prose safe to embed in a copy-paste prompt.

    Whitespace and newlines are collapsed, image beacons and unvetted links are
    removed, secret values are masked, and prompt-injection phrases and shell
    pipelines are replaced with [SANITIZED]. Ordinary technical vocabulary
    (eval, import, O(N^2), environment variable names) is left alone, and the
    text is never rejected: it is rewritten and, if too long, cut in place.
    """
    text = " ".join(str(text if text is not None else "").split())
    if not text:
        return ""
    text = _MARKDOWN_IMAGE.sub("[image removed]", text)
    for pattern in _SHELL_PATTERNS:
        text = pattern.sub(SANITIZED, text)
    for pattern in _INJECTION_PATTERNS:
        text = pattern.sub(SANITIZED, text)
    text = _URL.sub(_vet_url, text)
    text = mask_secrets(text)
    if len(text) > max_length:
        text = text[: max_length - 3].rstrip() + "..."
    return text


def looks_like_injection(text: Any) -> bool:
    text = str(text or "")
    return any(p.search(text) for p in _INJECTION_PATTERNS) or any(p.search(text) for p in _SHELL_PATTERNS)


def strip_code_comments(code: Any, file_name: str = "") -> str:
    """Remove prose comments from a quoted code snippet (for the report)."""
    ext = extension_of(file_name) or ".js"
    stripped = strip_comments(str(code or ""), ext)
    lines = [line.rstrip() for line in stripped.split("\n")]
    return "\n".join(line for line in lines if line.strip())
