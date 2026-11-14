"""Run checkpoints, so a crashed agent pipeline resumes instead of restarting.

After each task's output passes its guardrail, the accepted output and the
whole run state (read ledger, findings registry, inserted discoveries) are
written to `.reporover/runs/<repo key>/checkpoint.json`. `--resume` restores
that state and runs only the tasks that had not finished.

A checkpoint is only resumed if the repository still produces the same
detector findings; if the code changed, the run starts from the beginning.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from repo_rover.read_ledger import RunContext

CHECKPOINT_ROOT = Path(".reporover") / "runs"
CHECKPOINT_VERSION = 1
# The synthesis task is never skipped: it is the last step and re-rendering
# the report from restored state is cheap and deterministic.
RESUMABLE_TASKS = ("scout", "inspector", "red_team", "optimizer")


def repo_key(target: str | os.PathLike) -> str:
    real = os.path.normcase(os.path.realpath(target))
    name = "".join(c if c.isalnum() or c in "-_" else "-" for c in Path(real).name)[:40] or "repo"
    return f"{name}-{hashlib.sha256(real.encode('utf-8')).hexdigest()[:12]}"


def detector_signature(detector_findings: dict[str, dict[str, Any]]) -> str:
    """Hash of what the deterministic pass found. It changes if the code does."""
    shape = [
        (finding_id, finding.get("slug"), [(h.get("file"), h.get("line")) for h in finding.get("hits", [])])
        for finding_id, finding in sorted(detector_findings.items())
    ]
    return hashlib.sha256(json.dumps(shape, sort_keys=True).encode("utf-8")).hexdigest()


class RunCheckpoint:
    def __init__(self, base_directory: str | os.PathLike, target: str | os.PathLike):
        self.directory = Path(base_directory) / CHECKPOINT_ROOT / repo_key(target)
        self.path = self.directory / "checkpoint.json"
        self.target = os.path.realpath(target)
        self._completed: dict[str, str] = {}
        self._signature = ""

    # --- reading -------------------------------------------------------------
    def load(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict) or data.get("version") != CHECKPOINT_VERSION:
            return None
        return data

    def resume_into(self, ctx: RunContext) -> tuple[dict[str, str], str]:
        """Restore a crashed run into `ctx`.

        Returns (outputs of the tasks that already finished, message). The
        dict is empty when there is nothing safe to resume.
        """
        self._signature = detector_signature(ctx.detector_findings)
        data = self.load()
        if data is None:
            return {}, "No checkpoint found; starting a new run."
        if data.get("state") == "completed":
            return {}, "The last run completed; starting a new run."
        if data.get("detector_signature") != self._signature:
            return {}, "The repository changed since the checkpoint; starting a new run."
        completed = {
            name: text
            for name, text in (data.get("completed_tasks") or {}).items()
            if name in RESUMABLE_TASKS and isinstance(text, str)
        }
        if not completed:
            return {}, "The checkpoint has no finished tasks; starting a new run."
        ctx.restore(data.get("run") or {})
        self._completed = dict(completed)
        return completed, "Resuming after: " + ", ".join(sorted(completed)) + "."

    # --- writing -------------------------------------------------------------
    def start(self, ctx: RunContext) -> None:
        """Begin a new checkpoint file for a fresh run."""
        self._completed = {}
        self._signature = detector_signature(ctx.detector_findings)
        self._write(ctx, "running")

    def save_task(self, task_name: str, output_text: str, ctx: RunContext) -> None:
        """Record a task whose output passed its guardrail."""
        self._completed[task_name] = output_text
        self._write(ctx, "running")

    def mark_completed(self, ctx: RunContext) -> None:
        self._write(ctx, "completed")

    def _write(self, ctx: RunContext, state: str) -> None:
        payload = {
            "version": CHECKPOINT_VERSION,
            "state": state,
            "target": self.target,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "detector_signature": self._signature,
            "completed_tasks": self._completed,
            "run": ctx.snapshot(),
        }
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, default=str), encoding="utf-8")
            os.replace(temporary, self.path)  # atomic: a crash never leaves half a file
        except OSError as exc:
            # A checkpoint is a convenience; failing to write one must not stop a run.
            print(f"Warning: could not write checkpoint: {exc}")
