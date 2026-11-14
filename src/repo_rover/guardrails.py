"""Task guardrails that check agent output against the run's read ledger.

A guardrail is a callable CrewAI invokes with a task's output. Returning
(False, message) rejects the output and makes the agent retry with the message
as feedback. After `max_retries` rejections the guardrail stops rejecting:
it removes every claim it could not verify, marks the run as degraded (which
disables downstream database writes) and lets the pipeline continue.
"""

from __future__ import annotations

import json
import re
from typing import Any

from repo_rover.context_budget import fit_summary
from repo_rover.read_ledger import RunContext
from repo_rover.sanitize import sanitize_text

DEFAULT_MAX_RETRIES = 2
MIN_SCOUT_EVIDENCE = 2
MAX_PROBLEMS_REPORTED = 12
DEGRADED_KEY = "_reporover_degraded"


def extract_json_object(raw: Any) -> dict[str, Any] | None:
    """Parse the first JSON object in an LLM answer, tolerating code fences
    and surrounding prose. Returns None if there is no valid object."""
    text = str(raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        start = candidate.find("{")
        while start != -1:
            try:
                value, _ = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                start = candidate.find("{", start + 1)
                continue
            if isinstance(value, dict):
                return value
            start = candidate.find("{", start + 1)
    return None


def parse_line_range(value: Any) -> tuple[int, int] | None:
    """'42-57', '42', 'L42-L57', 'lines 42 to 57' -> (42, 57)."""
    numbers = [int(n) for n in re.findall(r"\d+", str(value or ""))]
    if not numbers:
        return None
    start, end = numbers[0], numbers[-1]
    return (start, end) if end >= start else (end, start)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


class LedgerGuardrail:
    """Base guardrail: parse JSON, verify claims, retry, then degrade."""

    agent_name = "Agent"
    task_name = ""  # checkpoint key; empty means the task is not checkpointed

    def __init__(self, ctx: RunContext, max_retries: int = DEFAULT_MAX_RETRIES) -> None:
        self.ctx = ctx
        self.max_retries = max_retries
        self.rejections = 0

    @property
    def crew_max_retries(self) -> int:
        """Value for Task.guardrail_max_retries: one more than this guardrail
        ever uses, so CrewAI's own retry-exhausted exception is never raised."""
        return self.max_retries + 1

    def as_task_guardrail(self):
        """The plain function to pass as Task(guardrail=...).

        CrewAI calls inspect.getsource() on the guardrail when it emits events,
        which fails for a callable object, so tasks get this function instead.
        """

        def ledger_guardrail(output):
            return self(output)

        return ledger_guardrail

    def verify(self, data: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
        """Return (problems, copy of data with unverifiable claims removed)."""
        raise NotImplementedError

    def fallback(self) -> dict[str, Any]:
        """Output to pass downstream when the agent never produced valid JSON."""
        raise NotImplementedError

    # No return annotation on purpose: this module uses postponed annotations,
    # and CrewAI rejects a guardrail whose annotation is a string.
    def __call__(self, output: Any):
        """Return (True, verified JSON text) or (False, feedback for a retry)."""
        data = extract_json_object(getattr(output, "raw", output))
        if data is None:
            problems = ["The answer is not a single valid JSON object."]
            cleaned = self.fallback()
        else:
            problems, cleaned = self.verify(data)

        if not problems:
            return True, self.finish(cleaned)

        if self.rejections < self.max_retries:
            self.rejections += 1
            self.ctx.artifacts.setdefault("guardrail_rejections", []).append(
                {"agent": self.agent_name, "problems": len(problems)}
            )
            return False, self._feedback(problems)

        reason = (
            f"{self.agent_name} output failed evidence validation after "
            f"{self.max_retries} retries; {len(problems)} unverified claim(s) removed."
        )
        self.ctx.mark_degraded(reason)
        cleaned[DEGRADED_KEY] = {"reason": reason, "removed_claims": problems}
        return True, self.finish(cleaned)

    def finish(self, cleaned: dict[str, Any]) -> str:
        """Record the accepted output on the run, then render what later agents see."""
        self.on_accept(cleaned)
        rendered = self.render(cleaned)
        checkpoint = getattr(self.ctx, "checkpoint", None)
        if checkpoint is not None and self.task_name:
            checkpoint.save_task(self.task_name, rendered, self.ctx)
        return rendered

    def on_accept(self, cleaned: dict[str, Any]) -> None:
        """Hook: called once with the verified output (after any stripping)."""

    def render(self, cleaned: dict[str, Any]) -> str:
        """The task output passed downstream, cut to the agent-summary budget."""
        return fit_summary(cleaned)

    def _feedback(self, problems: list[str]) -> str:
        shown = problems[:MAX_PROBLEMS_REPORTED]
        extra = len(problems) - len(shown)
        lines = [f"- {p}" for p in shown]
        if extra > 0:
            lines.append(f"- ...and {extra} more.")
        read = sorted(e.path for e in self.ctx.entries.values())
        return (
            "REJECTED: the answer contains claims that are not backed by the read "
            "ledger.\n" + "\n".join(lines) + "\n"
            "Files you have actually read in this run: "
            + (", ".join(read) if read else "none")
            + ".\nFix every item: either read the file with the repository tools "
            "and copy the quote character-for-character from the tool output, or "
            "delete the claim. Then return the complete corrected JSON object."
        )


class ScoutGuardrail(LedgerGuardrail):
    """Scout: evidence quotes must be verbatim from files read; key_files and
    entry_points must exist in the repository listing."""

    agent_name = "Scout"
    task_name = "scout"

    def fallback(self) -> dict[str, Any]:
        return {
            "repository_path": str(self.ctx.root),
            "primary_language": "unknown",
            "framework": "unknown",
            "stack_components": [],
            "business_intent": "unknown",
            "critical_user_flows": [],
            "sensitive_data": [],
            "entry_points": [],
            "key_files": [],
            "evidence": [],
        }

    def verify(self, data: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
        problems: list[str] = []
        cleaned = dict(data)

        if not isinstance(data.get("evidence"), list):
            problems.append('"evidence" must be an array of {file, lines, quote, fact} objects.')
        verified_evidence = []
        for index, item in enumerate(_as_list(data.get("evidence"))):
            label = f"evidence[{index}]"
            if not isinstance(item, dict):
                problems.append(f"{label} must be an object with file, lines, quote and fact.")
                continue
            file = item.get("file")
            if not self.ctx.has_read(file):
                problems.append(f"{label}: file '{file}' was never read in this run.")
                continue
            ok, why = self.ctx.quote_matches(file, item.get("quote"))
            if not ok:
                problems.append(f"{label} ({file}): {why}.")
                continue
            verified_evidence.append(item)
        cleaned["evidence"] = verified_evidence
        if isinstance(data.get("evidence"), list) and len(verified_evidence) < MIN_SCOUT_EVIDENCE:
            problems.append(
                f"At least {MIN_SCOUT_EVIDENCE} evidence items with verified quotes "
                f"are required; {len(verified_evidence)} verified."
            )

        for field_name in ("key_files", "entry_points"):
            kept = []
            for path in _as_list(data.get(field_name)):
                if self.ctx.in_listing(path):
                    kept.append(path)
                else:
                    problems.append(
                        f"{field_name}: '{path}' is not a file in the repository listing."
                    )
            cleaned[field_name] = kept
        return problems, cleaned

    def on_accept(self, cleaned: dict[str, Any]) -> None:
        fingerprint = self.ctx.fingerprint
        if fingerprint is not None:
            cleaned["primary_language"] = fingerprint.primary_language
            cleaned["framework"] = fingerprint.primary_framework
            cleaned["stack_components"] = fingerprint.component_names()
        self.ctx.artifacts["scout"] = cleaned


class OptimizerGuardrail(LedgerGuardrail):
    """Optimizer: every reviewed file must be in the ledger, and every finding
    must quote served code from line ranges that were actually served."""

    agent_name = "Optimizer"
    task_name = "optimizer"

    def fallback(self) -> dict[str, Any]:
        return {
            "files_reviewed": [],
            "findings": [],
            "bloat": [],
            "summary": "No verified optimizer findings were produced in this run.",
        }

    def _check_lines(self, label: str, file: Any, lines: Any, required: bool) -> str | None:
        parsed = parse_line_range(lines)
        if parsed is None:
            return f"{label}: \"lines\" must be a line range like \"42-57\"." if required else None
        if not self.ctx.lines_were_served(file, *parsed):
            return (
                f"{label} ({file}): lines {parsed[0]}-{parsed[1]} were not all "
                "served by the read tools in this run."
            )
        return None

    def verify(self, data: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
        problems: list[str] = []
        cleaned = dict(data)

        reviewed = []
        for path in _as_list(data.get("files_reviewed")):
            if self.ctx.has_read(path):
                reviewed.append(path)
            else:
                problems.append(f"files_reviewed: '{path}' was never read in this run.")
        cleaned["files_reviewed"] = reviewed

        findings = []
        for index, item in enumerate(_as_list(data.get("findings"))):
            label = f"findings[{index}]"
            if not isinstance(item, dict):
                problems.append(f"{label} must be an object.")
                continue
            file = item.get("file")
            if not self.ctx.has_read(file):
                problems.append(f"{label}: file '{file}' was never read in this run.")
                continue
            ok, why = self.ctx.quote_matches(file, item.get("offending_code"))
            if not ok:
                problems.append(f"{label} ({file}): offending_code - {why}.")
                continue
            line_problem = self._check_lines(label, file, item.get("lines"), required=True)
            if line_problem:
                problems.append(line_problem)
                continue
            findings.append(item)
        cleaned["findings"] = findings

        bloat = []
        for index, item in enumerate(_as_list(data.get("bloat"))):
            label = f"bloat[{index}]"
            if not isinstance(item, dict):
                problems.append(f"{label} must be an object.")
                continue
            file = item.get("file")
            if not self.ctx.has_read(file):
                problems.append(f"{label}: file '{file}' was never read in this run.")
                continue
            line_problem = self._check_lines(label, file, item.get("lines"), required=False)
            if line_problem:
                problems.append(line_problem)
                continue
            bloat.append(item)
        cleaned["bloat"] = bloat
        return problems, cleaned

    def on_accept(self, cleaned: dict[str, Any]) -> None:
        self.ctx.drop_findings("optimizer")
        for number, item in enumerate(cleaned.get("findings") or [], start=1):
            item["id"] = f"V{number}"
            item["label"] = "Verified"
            self.ctx.register_finding(
                {
                    "id": item["id"],
                    "label": "Verified",
                    "source": "optimizer",
                    "title": sanitize_text(item.get("issue") or "Inefficient code", 140),
                    "severity": str(item.get("impact") or "medium").lower(),
                    "files": [f"{item.get('file')}:{item.get('lines')}"],
                    "evidence": [
                        {
                            "kind": "quote",
                            "file": item.get("file"),
                            "lines": str(item.get("lines") or ""),
                            "quote": str(item.get("offending_code") or ""),
                        }
                    ],
                    "problem": sanitize_text(
                        f"{item.get('issue', '')} Current complexity: "
                        f"{item.get('current_complexity', 'unknown')}; target: "
                        f"{item.get('target_complexity', 'unknown')}.", 700
                    ),
                    "default_fix": sanitize_text(item.get("rewrite"), 900),
                    "default_checks": [],
                }
            )
