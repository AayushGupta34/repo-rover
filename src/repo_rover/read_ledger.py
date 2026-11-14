"""Per-run read ledger and run state.

Every repository tool records what it served to the LLM here: the file path,
the line ranges, and the SHA-256 of the file. Guardrails check agent claims
against this ledger, and database write tools check the run state before
writing. Nothing in this module talks to the network or the database.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_LINE_BUDGET = 4000
MIN_QUOTE_CHARS = 10

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_DEGRADED = "degraded"
STATUS_FAILED = "failed"

_LINE_NUMBER_PREFIX = re.compile(r"^\s*\d+\s*\|\s?")
_FENCE_LINE = re.compile(r"^`{3,}[\w+-]*$")


def _split_lines(text: Any) -> list[str]:
    return str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")


def normalize_source_lines(lines: list[str]) -> str:
    """Whitespace-normalize served file lines: strip each line, drop blank ones."""
    return "\n".join(s for s in (line.strip() for line in lines) if s)


def normalize_quote(quote: Any) -> str:
    """Whitespace-normalize an agent's quote the same way as served content.

    Line endings are unified, each line is stripped, blank lines are dropped.
    The `  12| ` line-number prefixes the read tool prints and markdown code
    fences are removed, since agents often copy them along with the code.
    """
    cleaned = []
    for raw in _split_lines(quote):
        line = _LINE_NUMBER_PREFIX.sub("", raw).strip()
        if line and not _FENCE_LINE.match(line):
            cleaned.append(line)
    return "\n".join(cleaned)


@dataclass
class LedgerEntry:
    path: str
    sha256: str
    total_lines: int
    lines: dict[int, str] = field(default_factory=dict)
    tools: set[str] = field(default_factory=set)

    def ranges(self) -> list[tuple[int, int]]:
        """Served line numbers merged into contiguous inclusive ranges."""
        merged: list[tuple[int, int]] = []
        for number in sorted(self.lines):
            if merged and number == merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], number)
            else:
                merged.append((number, number))
        return merged

    def segments(self) -> list[str]:
        """Normalized text of each contiguous served range."""
        return [
            normalize_source_lines([self.lines[n] for n in range(start, end + 1)])
            for start, end in self.ranges()
        ]


class RunContext:
    """State of one analysis run: the read ledger plus the degraded flag."""

    def __init__(self, target_root: str | os.PathLike, line_budget: int = RUN_LINE_BUDGET):
        self.root = Path(os.path.realpath(target_root))
        self.run_uuid = uuid.uuid4().hex
        self.started_at = datetime.now(timezone.utc)
        self.line_budget = line_budget
        self.lines_served = 0
        self.status = STATUS_RUNNING
        self.degraded_reasons: list[str] = []
        self.db_run_id: int | None = None
        self.listing: dict[str, str] | None = None
        self.entries: dict[str, LedgerEntry] = {}
        # Phase 3 evidence-first state.
        self.fingerprint: Any = None  # fingerprint.Fingerprint, set before the crew runs
        self.detector_findings: dict[str, dict[str, Any]] = {}  # "D1" -> detector finding
        self.findings: dict[str, dict[str, Any]] = {}  # report registry: id -> labelled finding
        self.inserted_discoveries: dict[int, dict[str, Any]] = {}  # rows this run wrote
        self.artifacts: dict[str, Any] = {}  # verified agent outputs, e.g. "scout"
        self.checkpoint: Any = None  # checkpoint.RunCheckpoint, when checkpointing is on
        self._lock = threading.Lock()

    # --- path handling -------------------------------------------------------
    def key(self, claimed_path: Any) -> str | None:
        """Canonical ledger key for a path an agent or tool names, or None if
        the path is empty or points outside the target directory."""
        text = str(claimed_path or "").strip().strip("`'\"").replace("\\", "/")
        if not text:
            return None
        try:
            if os.path.isabs(text):
                text = os.path.relpath(os.path.realpath(text), self.root)
            normalized = os.path.normpath(text).replace("\\", "/")
        except (ValueError, OSError):
            return None
        if normalized in ("", ".") or normalized == ".." or normalized.startswith("../"):
            return None
        return os.path.normcase(normalized).replace("\\", "/")

    # --- recording -----------------------------------------------------------
    def record_listing(self, relative_paths: list[str]) -> None:
        with self._lock:
            if self.listing is None:
                self.listing = {}
            for path in relative_paths:
                key = self.key(path)
                if key:
                    self.listing[key] = path

    def record_read(
        self,
        relative_path: str,
        sha256: str,
        total_lines: int,
        served: dict[int, str],
        tool: str,
        count_budget: bool = True,
    ) -> None:
        """Record lines served to the LLM. `count_budget=False` is for content
        the pipeline itself placed in a prompt (detector hits), which must be
        quotable but should not use up the agents' read budget."""
        key = self.key(relative_path)
        if key is None or not served:
            return
        with self._lock:
            entry = self.entries.get(key)
            if entry is None or entry.sha256 != sha256:
                # First read, or the file changed on disk: older lines are stale.
                entry = LedgerEntry(relative_path, sha256, total_lines)
                self.entries[key] = entry
            entry.lines.update(served)
            entry.tools.add(tool)
            if count_budget:
                self.lines_served += len(served)

    def budget_remaining(self) -> int:
        return max(0, self.line_budget - self.lines_served)

    # --- verification --------------------------------------------------------
    def entry_for(self, claimed_path: Any) -> LedgerEntry | None:
        key = self.key(claimed_path)
        return self.entries.get(key) if key else None

    def has_read(self, claimed_path: Any) -> bool:
        return self.entry_for(claimed_path) is not None

    def in_listing(self, claimed_path: Any) -> bool:
        key = self.key(claimed_path)
        if key is None:
            return False
        return key in self.entries or (self.listing is not None and key in self.listing)

    def lines_were_served(self, claimed_path: Any, start: int, end: int) -> bool:
        entry = self.entry_for(claimed_path)
        if entry is None or start < 1 or end < start:
            return False
        return all(number in entry.lines for number in range(start, end + 1))

    def quote_matches(self, claimed_path: Any, quote: Any) -> tuple[bool, str]:
        """True if the whitespace-normalized quote appears in one contiguous
        range of content this run actually served from that file."""
        entry = self.entry_for(claimed_path)
        if entry is None:
            return False, "file is not in the read ledger"
        needle = normalize_quote(quote)
        if len(re.sub(r"\s", "", needle)) < MIN_QUOTE_CHARS:
            return False, f"quote is shorter than {MIN_QUOTE_CHARS} non-space characters"
        if any(needle in segment for segment in entry.segments()):
            return True, ""
        return False, "quote does not match the content served from this file"

    # --- evidence ------------------------------------------------------------
    def find_detector(self, reference: Any) -> dict[str, Any] | None:
        """Look up a detector finding by its id ("D3") or catalogue slug."""
        text = str(reference or "").strip()
        if not text:
            return None
        if text.upper() in self.detector_findings:
            return self.detector_findings[text.upper()]
        lowered = text.lower()
        for finding in self.detector_findings.values():
            if finding.get("slug") == lowered:
                return finding
        return None

    def verify_evidence(self, evidence: Any) -> tuple[list[dict[str, Any]], list[str]]:
        """Check evidence items against this run.

        An item is verified if it either quotes content this run served
        ({"file", "quote"}, optional "lines") or references a detector finding
        from this run ({"detector": "D3" or a catalogue slug}).
        Returns (verified items, problems). Nothing is taken on trust.
        """
        verified: list[dict[str, Any]] = []
        problems: list[str] = []
        if not isinstance(evidence, list) or not evidence:
            return verified, ["evidence must be a non-empty list of objects"]
        for index, item in enumerate(evidence):
            label = f"evidence[{index}]"
            if not isinstance(item, dict):
                problems.append(f"{label}: must be an object")
                continue
            if item.get("detector"):
                finding = self.find_detector(item["detector"])
                if finding is None:
                    problems.append(
                        f"{label}: '{item['detector']}' is not a detector hit from this run"
                    )
                else:
                    verified.append(
                        {"kind": "detector", "detector": finding["slug"], "id": finding["id"]}
                    )
                continue
            file, quote = item.get("file"), item.get("quote")
            if not file or not quote:
                problems.append(f"{label}: needs either 'detector' or both 'file' and 'quote'")
                continue
            ok, why = self.quote_matches(file, quote)
            if not ok:
                problems.append(f"{label} ({file}): {why}")
                continue
            entry = self.entry_for(file)
            verified.append(
                {
                    "kind": "quote",
                    "file": entry.path if entry else str(file),
                    "lines": str(item.get("lines") or ""),
                    "quote": normalize_quote(quote),
                }
            )
        return verified, problems

    @staticmethod
    def evidence_label(verified: list[dict[str, Any]]) -> str:
        """Verified (an exact quote), Detector-Hit (a rule fired) or Hypothesis."""
        if any(item.get("kind") == "quote" for item in verified):
            return "Verified"
        if any(item.get("kind") == "detector" for item in verified):
            return "Detector-Hit"
        return "Hypothesis"

    def register_finding(self, finding: dict[str, Any]) -> None:
        with self._lock:
            self.findings[finding["id"]] = finding

    def drop_findings(self, source: str) -> None:
        with self._lock:
            for key in [k for k, f in self.findings.items() if f.get("source") == source]:
                del self.findings[key]

    # --- run state -----------------------------------------------------------
    def mark_degraded(self, reason: str) -> None:
        with self._lock:
            self.status = STATUS_DEGRADED
            self.degraded_reasons.append(reason)

    @property
    def is_degraded(self) -> bool:
        return self.status == STATUS_DEGRADED

    @property
    def writes_enabled(self) -> bool:
        return self.status == STATUS_RUNNING

    def final_status(self, failed: bool = False) -> str:
        if failed:
            return STATUS_FAILED
        return STATUS_DEGRADED if self.is_degraded else STATUS_COMPLETED

    # --- serialization -------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "run_uuid": self.run_uuid,
            "target_repo": str(self.root),
            "started_at": self.started_at.isoformat(),
            "status": self.status,
            "degraded_reasons": list(self.degraded_reasons),
            "files_listed": len(self.listing) if self.listing is not None else 0,
            "files_read": len(self.entries),
            "lines_served": self.lines_served,
            "line_budget": self.line_budget,
            "stack_trigger": getattr(self.fingerprint, "stack_trigger", None),
            "detector_findings": len(self.detector_findings),
            "report_findings": dict(
                sorted(
                    (label, sum(1 for f in self.findings.values() if f.get("label") == label))
                    for label in ("Verified", "Detector-Hit", "Hypothesis")
                )
            ),
            "discoveries_inserted": sorted(self.inserted_discoveries),
            "files": [
                {
                    "path": entry.path,
                    "sha256": entry.sha256,
                    "total_lines": entry.total_lines,
                    "line_ranges": [list(r) for r in entry.ranges()],
                    "tools": sorted(entry.tools),
                }
                for entry in sorted(self.entries.values(), key=lambda e: e.path)
            ],
        }

    def snapshot(self) -> dict[str, Any]:
        """Everything needed to continue this run in a new process.

        The fingerprint and detector findings are not included: they are
        deterministic and are recomputed when the run is resumed.
        """
        return {
            "run_uuid": self.run_uuid,
            "started_at": self.started_at.isoformat(),
            "lines_served": self.lines_served,
            "status": self.status,
            "degraded_reasons": list(self.degraded_reasons),
            "db_run_id": self.db_run_id,
            "listing": self.listing,
            "entries": [
                {
                    "path": entry.path,
                    "sha256": entry.sha256,
                    "total_lines": entry.total_lines,
                    "lines": {str(number): text for number, text in entry.lines.items()},
                    "tools": sorted(entry.tools),
                }
                for entry in self.entries.values()
            ],
            "findings": self.findings,
            "inserted_discoveries": {str(k): v for k, v in self.inserted_discoveries.items()},
            "artifacts": self.artifacts,
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Load a snapshot() into this run."""
        with self._lock:
            self.run_uuid = data.get("run_uuid") or self.run_uuid
            try:
                self.started_at = datetime.fromisoformat(data["started_at"])
            except (KeyError, TypeError, ValueError):
                pass
            self.lines_served = int(data.get("lines_served") or 0)
            self.status = data.get("status") or STATUS_RUNNING
            self.degraded_reasons = list(data.get("degraded_reasons") or [])
            self.db_run_id = data.get("db_run_id")
            self.listing = data.get("listing")
            self.entries = {}
            for item in data.get("entries") or []:
                entry = LedgerEntry(item["path"], item["sha256"], int(item["total_lines"]))
                entry.lines = {int(number): text for number, text in (item.get("lines") or {}).items()}
                entry.tools = set(item.get("tools") or [])
                key = self.key(entry.path)
                if key:
                    self.entries[key] = entry
            self.findings = dict(data.get("findings") or {})
            self.inserted_discoveries = {
                int(k): v for k, v in (data.get("inserted_discoveries") or {}).items()
            }
            self.artifacts = dict(data.get("artifacts") or {})

    def save(self, directory: str | os.PathLike) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        ledger_path = directory / "ledger.json"
        ledger_path.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        return ledger_path


_active_run: RunContext | None = None


def set_active_run(run: RunContext | None) -> None:
    global _active_run
    _active_run = run


def get_active_run() -> RunContext | None:
    return _active_run
