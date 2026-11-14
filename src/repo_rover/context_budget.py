"""Context budget helpers.

Everything handed to the LLM (tool output, detector hits, summaries of earlier
agents) is cut to a share of LLM_MAX_INPUT_TOKENS. Token counts use a
conservative character estimate: 1 token = 4 characters.
"""

from dotenv import load_dotenv

load_dotenv()

import json
import math
import os
from typing import Any

DEFAULT_MAX_INPUT_TOKENS = 6000
MIN_MAX_INPUT_TOKENS = 500
CHARS_PER_TOKEN = 4
TRUNCATION_MARKER = "\n[...truncated to fit the context budget...]"

# Shares of the input budget given to each kind of content.
TOOL_OUTPUT_SHARE = 0.40
DETECTOR_HITS_SHARE = 0.45
AGENT_SUMMARY_SHARE = 0.15
FINGERPRINT_SHARE = 0.10


def max_input_tokens() -> int:
    """LLM_MAX_INPUT_TOKENS from the environment, default 6000."""
    raw = os.getenv("LLM_MAX_INPUT_TOKENS", "").strip()
    try:
        value = int(raw) if raw else DEFAULT_MAX_INPUT_TOKENS
    except ValueError:
        value = DEFAULT_MAX_INPUT_TOKENS
    return max(MIN_MAX_INPUT_TOKENS, value)


def estimate_tokens(text: Any) -> int:
    return math.ceil(len(str(text or "")) / CHARS_PER_TOKEN)


def budget_tokens(share: float) -> int:
    """Token allowance for a share (0..1) of the input budget."""
    return max(50, int(max_input_tokens() * share))


def budget_chars(share: float) -> int:
    return budget_tokens(share) * CHARS_PER_TOKEN


def truncate_to_tokens(text: Any, max_tokens: int, marker: str = TRUNCATION_MARKER) -> str:
    """Cut text to at most max_tokens, on a line boundary where possible."""
    text = str(text or "")
    limit = max(0, max_tokens) * CHARS_PER_TOKEN
    if len(text) <= limit:
        return text
    keep = max(0, limit - len(marker))
    head = text[:keep]
    newline = head.rfind("\n")
    if newline >= keep * 0.6:
        head = head[:newline]
    return head.rstrip() + marker


def fit_tool_output(text: Any) -> str:
    return truncate_to_tokens(text, budget_tokens(TOOL_OUTPUT_SHARE))


def _cap_strings(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, list):
        return [_cap_strings(item, limit) for item in value]
    if isinstance(value, dict):
        return {key: _cap_strings(item, limit) for key, item in value.items()}
    return value


def _cap_lists(value: Any, limit: int) -> Any:
    if isinstance(value, list):
        kept = [_cap_lists(item, limit) for item in value[:limit]]
        if len(value) > limit:
            kept.append(f"...(+{len(value) - limit} more omitted)")
        return kept
    if isinstance(value, dict):
        return {key: _cap_lists(item, limit) for key, item in value.items()}
    return value


def shrink_json(value: Any, max_tokens: int) -> str:
    """Serialize a summary object compactly, trimming it until it fits.

    Long strings are shortened first, then long lists are cut from the end, so
    the result stays valid JSON. Only if that is still not enough is the text
    itself truncated.
    """
    def dump(obj: Any) -> str:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, default=str)

    text = dump(value)
    if estimate_tokens(text) <= max_tokens:
        return text
    current = value
    for string_cap in (400, 200, 120, 60):
        current = _cap_strings(current, string_cap)
        text = dump(current)
        if estimate_tokens(text) <= max_tokens:
            return text
    for list_cap in (20, 10, 5, 3, 1):
        current = _cap_lists(current, list_cap)
        text = dump(current)
        if estimate_tokens(text) <= max_tokens:
            return text
    return truncate_to_tokens(text, max_tokens)


def fit_summary(value: Any) -> str:
    """A prior-agent summary cut to its share of the input budget."""
    return shrink_json(value, budget_tokens(AGENT_SUMMARY_SHARE))
