"""`reporover chaos self`: fault injection into RepoRover's own LLM and
database paths.

Everything runs offline and in-process. The model is a stand-in object and the
database is an in-memory fake that implements the handful of statements
RepoRover issues, including ON CONFLICT DO NOTHING and transactions. Backoff
sleeps are recorded instead of slept, so the suite takes about a second.

Each experiment states a hypothesis about how RepoRover should cope with one
fault. A failed experiment means RepoRover has the weakness it reports in
other people's code.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psycopg2

from repo_rover import catalog, db_tools, resilience, runtime
from repo_rover.chaos.framework import (
    BlastRadius,
    ChaosContext,
    ChaosRunner,
    DataMonkey,
    Experiment,
    ExperimentResult,
    FaultSpec,
    NetworkMonkey,
    Probe,
    ResourceMonkey,
    summarize,
)
from repo_rover.checkpoint import RunCheckpoint
from repo_rover.detector_engine import run_detectors
from repo_rover.evidence_report import InspectorGuardrail, render_report
from repo_rover.fingerprint import fingerprint_repo
from repo_rover.guardrails import ScoutGuardrail, extract_json_object
from repo_rover.read_ledger import RunContext, get_active_run, set_active_run
from repo_rover.repo_tools import build_repo_tools

LLM_TARGET = "in-process:llm"
DB_TARGET = "in-process:database"
PIPELINE_TARGET = "in-process:pipeline"

FIXTURE_FILES = {
    "README.md": "# Demo Shop\n\nA small shop where customers place orders and pay online.\n",
    "package.json": '{\n  "name": "demo-shop",\n  "dependencies": {"express": "^4.19.0"}\n}\n',
    "server.js": (
        'const express = require("express");\n'
        "const app = express();\n"
        'app.get("/orders/:id", async (req, res) => {\n'
        "  const rows = await db.query(`SELECT * FROM orders WHERE id = ${req.params.id}`);\n"
        "  res.json(rows);\n"
        "});\n"
        "app.listen(3000);\n"
    ),
}


# --------------------------------------------------------------------------- #
# Stand-ins for the model and the database
# --------------------------------------------------------------------------- #
class FakeChat:
    """Stands in for a chat model client. `invoke` is the seam monkeys wrap."""

    def __init__(self, model_name: str, reply: str):
        self.model_name = model_name
        self.temperature = 0.1
        self.reply = reply
        self.calls = 0

    def invoke(self, messages, stop=None):
        self.calls += 1
        return SimpleNamespace(content=self.reply, usage_metadata=None)


class FakeDatabase:
    """In-memory stand-in for Postgres with transactions and the unique keys
    RepoRover relies on. Seams: `connect`, `run_statement`, `after_commit`."""

    def __init__(self):
        self.discoveries: dict[str, int] = {}   # dedupe_key -> id
        self.confidences: dict[str, float] = {}  # dedupe_key -> stored confidence
        self.runs: dict[str, int] = {}          # run_uuid -> id
        self.run_updates: list[tuple] = []
        self.connections = 0
        self.statements = 0
        self._next_id = 100

    def connect(self, dsn=None, connect_timeout=None):
        self.connections += 1
        return _FakeConnection(self)

    def after_commit(self) -> None:
        """Called once a commit is durable; a fault here is a lost acknowledgement."""

    def run_statement(self, sql: str, params: tuple, pending: list) -> Any:
        self.statements += 1
        text = " ".join(sql.split()).upper()
        if text.startswith("INSERT INTO DYNAMIC_DISCOVERIES"):
            key, confidence = params[7], float(params[4])
            if key in self.discoveries:
                # ON CONFLICT (dedupe_key) DO UPDATE ... WHERE EXCLUDED.confidence > stored
                if confidence > self.confidences.get(key, -1.0):
                    pending.append((("confidence", key), confidence))
                    return (self.discoveries[key], False)
                return None
            if any(p == ("discovery", key) for p, _ in pending):
                return None
            self._next_id += 1
            pending.append((("discovery", key), self._next_id))
            pending.append((("confidence", key), confidence))
            return (self._next_id, True)
        if text.startswith("SELECT ID FROM DYNAMIC_DISCOVERIES"):
            found = self.discoveries.get(params[0])
            return (found,) if found is not None else None
        if text.startswith("INSERT INTO ANALYSIS_RUNS"):
            key = params[0]
            if key in self.runs:
                return None  # ON CONFLICT (run_uuid) DO NOTHING
            self._next_id += 1
            pending.append((("run", key), self._next_id))
            return (self._next_id,)
        if text.startswith("SELECT ID FROM ANALYSIS_RUNS"):
            found = self.runs.get(params[0])
            return (found,) if found is not None else None
        if text.startswith("UPDATE ANALYSIS_RUNS"):
            pending.append((("update", params[-1]), params[0]))
            return None
        raise AssertionError(f"FakeDatabase does not implement: {text[:60]}")

    def apply(self, pending: list) -> None:
        for (kind, key), value in pending:
            if kind == "discovery":
                self.discoveries[key] = value
            elif kind == "confidence":
                self.confidences[key] = value
            elif kind == "run":
                self.runs[key] = value
            else:
                self.run_updates.append((key, value))


class _FakeConnection:
    def __init__(self, database: FakeDatabase):
        self.database = database
        self.pending: list = []

    def cursor(self, cursor_factory=None):
        return _FakeCursor(self)

    def commit(self):
        self.database.apply(self.pending)
        self.pending = []
        self.database.after_commit()

    def rollback(self):
        self.pending = []

    def close(self):
        pass


class _FakeCursor:
    def __init__(self, connection: _FakeConnection):
        self.connection = connection
        self.result: Any = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        if " ".join(sql.split()).upper().startswith("SET LOCAL"):
            self.result = None
            return
        self.result = self.connection.database.run_statement(sql, tuple(params), self.connection.pending)

    def fetchone(self):
        return self.result

    def fetchall(self):
        return [] if self.result is None else [self.result]


class _Environment:
    """Patches installed for the whole suite and restored afterwards."""

    def __init__(self, seed: int):
        self.seed = seed
        self.sleeps: list[float] = []
        self.database = FakeDatabase()
        self.workdir = Path(tempfile.mkdtemp(prefix="reporover_chaos_"))
        self.repo = self.workdir / "repo"
        for relative, content in FIXTURE_FILES.items():
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
        self.catalog_entries = catalog.load_catalog(catalog.catalog_path())["entries"]
        self._saved: dict[str, Any] = {}

    def install(self) -> None:
        self._saved = {
            "sleep": resilience.SLEEP,
            "rng": resilience.RNG,
            "psycopg2": db_tools.psycopg2,
            "database_url": os.environ.get("DATABASE_URL"),
            "run": get_active_run(),
        }
        resilience.SLEEP = self.sleeps.append
        resilience.RNG = random.Random(self.seed)
        env = self
        db_tools.psycopg2 = SimpleNamespace(
            connect=lambda *args, **kwargs: env.database.connect(*args, **kwargs)
        )
        os.environ["DATABASE_URL"] = "postgresql://chaos:chaos@in-process/fake"
        set_active_run(None)

    def restore(self) -> None:
        resilience.SLEEP = self._saved["sleep"]
        resilience.RNG = self._saved["rng"]
        db_tools.psycopg2 = self._saved["psycopg2"]
        if self._saved["database_url"] is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = self._saved["database_url"]
        set_active_run(self._saved["run"])
        shutil.rmtree(self.workdir, ignore_errors=True)

    def reset(self) -> bool:
        """First steady-state probe of every experiment: a clean slate."""
        self.database = FakeDatabase()
        self.sleeps.clear()
        resilience.RNG = random.Random(self.seed)
        set_active_run(None)
        return True

    def analysed_run(self) -> RunContext:
        ctx = RunContext(self.repo)
        ctx.fingerprint = fingerprint_repo(self.repo)
        run_detectors(self.repo, self.catalog_entries, ctx.fingerprint, ctx)
        return ctx


def _discovery_args(suffix: str = "") -> dict[str, Any]:
    return dict(
        vulnerability_type="cache_poisoning" + suffix,
        stack_trigger="next 14.x",
        binary_flags_dict={"security_impact": True},
        fix_recommendation="Upgrade to the patched release.",
        confidence=0.7,
        evidence=[{"source_url": "https://example.com/advisory", "quote": "Affected versions allow cache poisoning."}],
    )


def _jitter_ok(sleeps: list[float], base: float, cap: float) -> bool:
    return all(0.0 <= delay <= min(cap, base * (2 ** i)) + 1e-9 for i, delay in enumerate(sleeps))


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def build_experiments(env: _Environment) -> list[Experiment]:
    experiments: list[Experiment] = []
    reset = Probe("clean environment", env.reset)

    def llm_pair():
        primary = FakeChat("llama-3.3-70b-versatile", "primary answer")
        smaller = FakeChat("llama-3.1-8b-instant", "fallback answer")
        return primary, smaller, runtime.GroqChatLLM(primary, fallbacks=[smaller])

    def llm_probe(llm, primary):
        def check():
            before = primary.calls
            ok = llm.call("health check") == "primary answer"
            return ok and primary.calls == before + 1
        return Probe("model answers on the first attempt when healthy", check)

    def db_probe():
        return Probe(
            "database records a run when healthy",
            lambda: db_tools.start_analysis_run(uuid.uuid4().hex, "probe") is not None,
        )

    llm_radius = BlastRadius([LLM_TARGET], max_duration_seconds=20)
    db_radius = BlastRadius([DB_TARGET], max_duration_seconds=20)
    pipeline_radius = BlastRadius([PIPELINE_TARGET, LLM_TARGET, DB_TARGET], max_duration_seconds=30)

    # --- S01: burst of 429s ---------------------------------------------------
    primary, smaller, llm = llm_pair()

    s01_monkey = NetworkMonkey(primary, "invoke", mode="http_error", status_code=429, failures=3)

    def s01_action(ctx: ChaosContext, llm=llm, smaller=smaller):
        env.sleeps.clear()
        before = s01_monkey.calls
        text = llm.call("summarise")
        return {"text": text, "sleeps": list(env.sleeps),
                "primary_calls": s01_monkey.calls - before, "fallback_calls": smaller.calls}

    def s01_evaluate(obs, ctx):
        if isinstance(obs, BaseException):
            return False, f"call raised {type(obs).__name__}: {obs}"
        ok = (obs["text"] == "primary answer" and obs["primary_calls"] == 4 and obs["fallback_calls"] == 0
              and len(obs["sleeps"]) == 3
              and _jitter_ok(obs["sleeps"], runtime.LLM_BASE_DELAY_SECONDS, runtime.LLM_MAX_DELAY_SECONDS))
        return ok, f"answered after {len(obs['sleeps'])} backoff waits {[round(s, 2) for s in obs['sleeps']]} without using the fallback"

    experiments.append(Experiment(
        id="S01", name="LLM returns HTTP 429 three times",
        hypothesis="A burst of rate-limit responses is absorbed by exponential backoff with full jitter; the primary model still answers.",
        fault=FaultSpec("http_error", {"status": 429, "failures": 3}), blast_radius=llm_radius, targets=[LLM_TARGET],
        steady_state=[reset, llm_probe(llm, primary)],
        monkeys=[s01_monkey],
        action=s01_action, evaluate=s01_evaluate,
    ))

    # --- S02: primary model down with 503 -------------------------------------
    primary, smaller, llm = llm_pair()

    def s02_action(ctx, llm=llm, primary=primary, smaller=smaller):
        primary.calls = 0
        monkey_calls_before = s02_monkey.calls
        text = llm.call("summarise")
        return {"text": text, "primary_attempts": s02_monkey.calls - monkey_calls_before,
                "fallback_calls": smaller.calls, "events": list(llm.resilience_events)}

    def s02_evaluate(obs, ctx):
        if isinstance(obs, BaseException):
            return False, f"call raised {type(obs).__name__}: {obs}"
        fell_back = any(e["event"] == "fallback" for e in obs["events"])
        ok = (obs["text"] == "fallback answer" and fell_back
              and obs["primary_attempts"] == runtime.LLM_MAX_ATTEMPTS and obs["fallback_calls"] == 1)
        return ok, f"primary tried {obs['primary_attempts']} times, then the smaller model answered"

    s02_monkey = NetworkMonkey(primary, "invoke", mode="http_error", status_code=503)
    experiments.append(Experiment(
        id="S02", name="Primary model answers 503 on every call",
        hypothesis="When the primary model stays unavailable, the fallback chain switches to the smaller model and the call succeeds.",
        fault=FaultSpec("http_error", {"status": 503, "failures": "all"}), blast_radius=llm_radius, targets=[LLM_TARGET],
        steady_state=[reset, llm_probe(llm, primary)], monkeys=[s02_monkey],
        action=s02_action, evaluate=s02_evaluate,
    ))

    # --- S03: every model offline ---------------------------------------------
    primary, smaller, llm = llm_pair()

    def s03_action(ctx, llm=llm):
        run = env.analysed_run()
        try:
            llm.call("judge the hits")
            error = None
        except Exception as exc:
            error = exc
        if not any(f.get("source") == "catalog" for f in run.findings.values()):
            InspectorGuardrail(run).on_accept({"judgments": []})
        return {"error": error, "report": render_report(run, None), "sleeps": len(env.sleeps)}

    def s03_evaluate(obs, ctx):
        if isinstance(obs, BaseException):
            return False, f"pipeline raised {type(obs).__name__}: {obs}"
        ok = (isinstance(obs["error"], runtime.LLMUnavailableError)
              and "[Detector-Hit] SQL assembled with string interpolation" in obs["report"]
              and "# Hypotheses (Unverified)" in obs["report"])
        return ok, "model outage raised a clean LLMUnavailableError; the deterministic report was still produced"

    experiments.append(Experiment(
        id="S03", name="Network offline for every model",
        hypothesis="A total LLM outage fails fast with one clear error, and the run still yields a report from the deterministic detector hits.",
        fault=FaultSpec("offline", {"models": "all"}), blast_radius=pipeline_radius, targets=[LLM_TARGET, PIPELINE_TARGET],
        steady_state=[reset, llm_probe(llm, primary)],
        monkeys=[NetworkMonkey(primary, "invoke", mode="offline"), NetworkMonkey(smaller, "invoke", mode="offline")],
        action=s03_action, evaluate=s03_evaluate,
    ))

    # --- S04: slow responses ---------------------------------------------------
    primary, smaller, llm = llm_pair()
    latency_log: list[float] = []

    def s04_action(ctx, llm=llm, smaller=smaller):
        latency_log.clear()
        return {"text": llm.call("summarise"), "latency": list(latency_log),
                "retries": len(env.sleeps), "fallback_calls": smaller.calls}

    def s04_evaluate(obs, ctx):
        if isinstance(obs, BaseException):
            return False, f"call raised {type(obs).__name__}: {obs}"
        ok = obs["text"] == "primary answer" and obs["retries"] == 0 and obs["fallback_calls"] == 0 and len(obs["latency"]) == 1
        return ok, f"tolerated {obs['latency'][0]:.1f}s of injected latency with no retry or fallback"

    experiments.append(Experiment(
        id="S04", name="LLM responses delayed by seconds",
        hypothesis="Slow but successful responses are accepted as they are: no retry storm and no needless fallback.",
        fault=FaultSpec("latency", {"seconds": 8}), blast_radius=llm_radius, targets=[LLM_TARGET],
        steady_state=[reset, llm_probe(llm, primary)],
        monkeys=[NetworkMonkey(primary, "invoke", mode="latency", latency_seconds=8, sleep=latency_log.append)],
        action=s04_action, evaluate=s04_evaluate,
    ))

    # --- S05: malformed JSON completions ---------------------------------------
    primary, smaller, llm = llm_pair()
    samples = [s for s in DataMonkey(seed=env.seed).malformed_json_samples() if s.strip()]
    sample_iter = {"n": 0}

    def corrupt(value, rng):
        text = samples[sample_iter["n"] % len(samples)]
        sample_iter["n"] += 1
        return SimpleNamespace(content=text, usage_metadata=None)

    def s05_action(ctx, llm=llm):
        run = env.analysed_run()
        set_active_run(run)
        guardrail = InspectorGuardrail(run, max_retries=2)
        verdicts = []
        for _ in range(3):
            completion = llm.call("return the judgments JSON")
            accepted, _payload = guardrail(SimpleNamespace(raw=completion))
            verdicts.append(accepted)
        build_repo_tools(run)["read"].run(file_path="README.md")
        quoted = json.loads(db_tools.InsertDiscoveryTool.run(**{
            **_discovery_args("_quoted"), "stack_trigger": "Express",
            "evidence": [{"file": "README.md", "quote": "A small shop where customers place orders"}]}))
        deterministic = json.loads(db_tools.InsertDiscoveryTool.run(**{
            **_discovery_args(), "stack_trigger": "Express", "evidence": [{"detector": "D1"}]}))
        return {"verdicts": verdicts, "degraded": run.is_degraded, "quoted": quoted, "deterministic": deterministic,
                "sources": {f["source"] for f in run.findings.values()}, "rows": len(env.database.discoveries)}

    def s05_evaluate(obs, ctx):
        if isinstance(obs, BaseException):
            return False, f"pipeline raised {type(obs).__name__}: {obs}"
        ok = (obs["verdicts"] == [False, False, True] and obs["degraded"] and obs["quoted"].get("status") == "skipped"
              and obs["deterministic"].get("status") == "inserted" and obs["rows"] == 1 and obs["sources"] == {"catalog"})
        return ok, ("two malformed completions were rejected for retry, the third degraded the run; a claim resting on an "
                    "agent quote was then blocked while a detector-backed finding was still recorded")

    experiments.append(Experiment(
        id="S05", name="LLM returns malformed JSON repeatedly",
        hypothesis="Unparseable completions never crash the pipeline: the guardrail retries, then degrades the run, keeps only detector findings and blocks database writes that depend on agent output.",
        fault=FaultSpec("malformed_output", {"samples": len(samples)}), blast_radius=pipeline_radius,
        targets=[LLM_TARGET, PIPELINE_TARGET], steady_state=[reset, llm_probe(llm, primary)],
        monkeys=[DataMonkey(primary, "invoke", corrupt=corrupt)],
        action=s05_action, evaluate=s05_evaluate,
    ))

    # --- S06: database statement timeouts --------------------------------------
    def timeout_error():
        return psycopg2.errors.QueryCanceled("canceling statement due to statement timeout")

    def s06_action(ctx):
        env.sleeps.clear()
        result = json.loads(db_tools.InsertDiscoveryTool.run(**_discovery_args()))
        return {"result": result, "rows": len(env.database.discoveries), "sleeps": list(env.sleeps)}

    def s06_evaluate(obs, ctx):
        if isinstance(obs, BaseException):
            return False, f"insert raised {type(obs).__name__}: {obs}"
        ok = (obs["result"].get("status") == "inserted" and obs["rows"] == 1 and len(obs["sleeps"]) == 2
              and _jitter_ok(obs["sleeps"], db_tools.DB_BASE_DELAY_SECONDS, db_tools.DB_MAX_DELAY_SECONDS))
        return ok, f"insert succeeded on the third attempt after waits {[round(s, 2) for s in obs['sleeps']]}; exactly one row written"

    def db_monkey(seam: str, **kwargs):
        """A monkey on a seam of whichever FakeDatabase is current at injection."""
        monkey = ResourceMonkey(None, seam, **kwargs)
        monkey.inject_original = monkey.inject

        def inject(ctx, monkey=monkey):
            monkey.owner = env.database  # the database created by this experiment's reset probe
            monkey.inject_original(ctx)

        monkey.inject = inject
        return monkey

    experiments.append(Experiment(
        id="S06", name="Database statement timeout on two attempts",
        hypothesis="A statement timeout is retried with backoff and full jitter, and the write lands exactly once.",
        fault=FaultSpec("statement_timeout", {"failures": 2}), blast_radius=db_radius, targets=[DB_TARGET],
        steady_state=[reset, db_probe()],
        monkeys=[db_monkey("run_statement", allow_calls=0, failures=2, error_factory=timeout_error)],
        action=s06_action, evaluate=s06_evaluate,
    ))

    # --- S07: database unreachable ----------------------------------------------
    def s07_action(ctx):
        env.sleeps.clear()
        before = env.database.connections
        run_id = db_tools.start_analysis_run(uuid.uuid4().hex, str(env.repo))
        insert = json.loads(db_tools.InsertDiscoveryTool.run(**_discovery_args()))
        finished = db_tools.finish_analysis_run(1, "failed", {}, [])
        return {"run_id": run_id, "insert": insert, "finished": finished,
                "attempts": s07_monkey.calls, "sleeps": len(env.sleeps), "rows": len(env.database.discoveries)}

    def s07_evaluate(obs, ctx):
        if isinstance(obs, BaseException):
            return False, f"database outage raised {type(obs).__name__}: {obs}"
        ok = (obs["run_id"] is None and "error" in obs["insert"] and obs["finished"] is False
              and obs["attempts"] == 3 * db_tools.DB_MAX_ATTEMPTS and obs["rows"] == 0)
        return ok, f"each of 3 operations gave up after {db_tools.DB_MAX_ATTEMPTS} attempts and reported failure without raising"

    s07_monkey = ResourceMonkey(None, "connect", allow_calls=0,
                                error_factory=lambda: psycopg2.OperationalError("connection refused"))
    s07_monkey.inject_original = s07_monkey.inject

    def s07_inject(ctx):
        s07_monkey.owner = env.database
        s07_monkey.inject_original(ctx)

    s07_monkey.inject = s07_inject
    experiments.append(Experiment(
        id="S07", name="Database unreachable",
        hypothesis="With the database down, run tracking is skipped and the insert tool returns an error message; nothing raises and the analysis can continue.",
        fault=FaultSpec("connection_refused", {"failures": "all"}), blast_radius=db_radius, targets=[DB_TARGET],
        steady_state=[reset, db_probe()], monkeys=[s07_monkey],
        action=s07_action, evaluate=s07_evaluate,
    ))

    # --- S08: commit succeeds but the acknowledgement is lost --------------------
    def s08_action(ctx):
        first = json.loads(db_tools.InsertDiscoveryTool.run(**_discovery_args()))
        repeat = json.loads(db_tools.InsertDiscoveryTool.run(**_discovery_args()))
        return {"first": first, "repeat": repeat, "rows": len(env.database.discoveries)}

    def s08_evaluate(obs, ctx):
        if isinstance(obs, BaseException):
            return False, f"insert raised {type(obs).__name__}: {obs}"
        ok = (obs["rows"] == 1 and obs["first"].get("status") == "already_recorded"
              and obs["repeat"].get("status") == "already_recorded" and obs["first"]["id"] == obs["repeat"]["id"])
        return ok, "the retried write and a later repeat both resolved to the same single row (ON CONFLICT DO NOTHING)"

    experiments.append(Experiment(
        id="S08", name="Connection drops after commit",
        hypothesis="If a write commits but its acknowledgement is lost, the retry does not create a duplicate row.",
        fault=FaultSpec("lost_acknowledgement", {"failures": 1}), blast_radius=db_radius, targets=[DB_TARGET],
        steady_state=[reset, db_probe()],
        monkeys=[db_monkey("after_commit", allow_calls=0, failures=1,
                           error_factory=lambda: psycopg2.OperationalError("server closed the connection unexpectedly"))],
        action=s08_action, evaluate=s08_evaluate,
    ))

    # --- S09: crash mid-pipeline, then resume ------------------------------------
    def s09_action(ctx):
        checkpoint_base = env.workdir / "checkpoints"
        first = env.analysed_run()
        checkpoint = RunCheckpoint(checkpoint_base, env.repo)
        checkpoint.start(first)
        first.checkpoint = checkpoint
        tools = build_repo_tools(first)
        tools["list"].run()
        tools["read"].run(file_path="README.md")
        ScoutGuardrail(first)(SimpleNamespace(raw=json.dumps({
            "business_intent": "Customers place orders and pay online.", "critical_user_flows": ["checkout"],
            "entry_points": ["server.js"], "key_files": ["server.js"],
            "evidence": [{"file": "README.md", "lines": "3", "quote": "A small shop where customers place orders and pay online.", "fact": "intent"},
                         {"file": "README.md", "lines": "3", "quote": "customers place orders and pay online", "fact": "payments"}]})))
        InspectorGuardrail(first)(SimpleNamespace(raw=json.dumps({"judgments": [
            {"id": "D1", "verdict": "applies", "severity": "critical", "reason": "Route parameter reaches SQL."}]})))
        expected = (first.run_uuid, sorted(first.findings), first.lines_served)
        del first  # the process "crashes" here, before the Red Teamer runs

        resumed = env.analysed_run()
        completed, message = RunCheckpoint(checkpoint_base, env.repo).resume_into(resumed)
        quote_ok, _ = resumed.quote_matches("README.md", "A small shop where customers place orders and pay online.")
        changed = env.analysed_run()
        (env.repo / "extra.js").write_text("eval(req.query.x)\n", encoding="utf-8")
        changed = env.analysed_run()
        stale, stale_message = RunCheckpoint(checkpoint_base, env.repo).resume_into(changed)
        (env.repo / "extra.js").unlink()
        return {"completed": sorted(completed), "state": (resumed.run_uuid, sorted(resumed.findings), resumed.lines_served),
                "expected": expected, "quote_ok": quote_ok, "scout": resumed.artifacts.get("scout", {}).get("business_intent"),
                "stale": stale, "stale_message": stale_message}

    def s09_evaluate(obs, ctx):
        if isinstance(obs, BaseException):
            return False, f"resume raised {type(obs).__name__}: {obs}"
        ok = (obs["completed"] == ["inspector", "scout"] and obs["state"] == obs["expected"] and obs["quote_ok"]
              and obs["scout"] == "Customers place orders and pay online." and obs["stale"] == {})
        return ok, "resumed with Scout and Inspector restored (same run id, findings and ledger); a changed repository correctly forced a fresh run"

    experiments.append(Experiment(
        id="S09", name="Process crashes after the Inspector",
        hypothesis="A crashed pipeline resumes from its checkpoint with finished tasks, findings and the read ledger intact, and refuses a checkpoint from changed code.",
        fault=FaultSpec("process_crash", {"after_task": "inspector"}), blast_radius=pipeline_radius, targets=[PIPELINE_TARGET],
        steady_state=[reset], action=s09_action, evaluate=s09_evaluate,
    ))

    # --- S10: read budget exhausted ------------------------------------------------
    def s10_action(ctx):
        run = RunContext(env.repo, line_budget=4)
        tools = build_repo_tools(run)
        outputs = [tools["read"].run(file_path="server.js"), tools["read"].run(file_path="README.md"),
                   tools["search"].run(query="express"), tools["outline"].run(file_path="server.js")]
        return {"outputs": outputs, "served": run.lines_served}

    def s10_evaluate(obs, ctx):
        if isinstance(obs, BaseException):
            return False, f"tool raised {type(obs).__name__}: {obs}"
        refused = [o for o in obs["outputs"][1:] if o.startswith("Error:") and "budget" in o]
        ok = obs["served"] == 4 and len(refused) == 3
        return ok, "after the budget was spent, every further read, search and outline was refused with an explanation"

    experiments.append(Experiment(
        id="S10", name="Read budget exhausted mid-run",
        hypothesis="When the context read budget runs out, tools refuse further reads with a clear message instead of failing or overserving.",
        fault=FaultSpec("quota_exhaustion", {"line_budget": 4}), blast_radius=pipeline_radius, targets=[PIPELINE_TARGET],
        steady_state=[reset], action=s10_action, evaluate=s10_evaluate,
    ))

    # --- S11: oversized and malformed inputs ----------------------------------------
    def s11_action(ctx):
        monkey = DataMonkey(seed=ctx.seed)
        run = env.analysed_run()
        set_active_run(run)
        tools = build_repo_tools(run)
        crashes, responses = [], 0
        calls = []
        for text in monkey.fuzz_strings():
            calls += [(tools["read"].run, {"file_path": text}), (tools["outline"].run, {"file_path": text}),
                      (tools["search"].run, {"query": text}), (tools["search"].run, {"query": text, "is_regex": True})]
        huge = monkey.oversized_text(600_000)
        for evidence in monkey.malformed_json_samples() + [huge, [{"file": "server.js", "quote": huge}], [{"detector": huge[:5000]}], [None, 7, "x"]]:
            calls.append((db_tools.InsertDiscoveryTool.run, {**_discovery_args(), "stack_trigger": "Express", "evidence": evidence}))
        calls.append((db_tools.InsertDiscoveryTool.run, {**_discovery_args(), "stack_trigger": huge[:20_000], "evidence": [{"detector": "D1"}]}))
        for function, kwargs in calls:
            try:
                function(**kwargs)
                responses += 1
            except ValueError as exc:
                if "arguments validation failed" in str(exc):
                    responses += 1  # rejected by the tool's argument schema
                else:
                    crashes.append(f"{type(exc).__name__}: {str(exc)[:80]}")
            except Exception as exc:
                crashes.append(f"{type(exc).__name__}: {str(exc)[:80]}")
        parsed = [extract_json_object(sample) for sample in monkey.malformed_json_samples()]
        return {"crashes": crashes, "calls": len(calls), "responses": responses,
                "rows": len(env.database.discoveries), "parsed_objects": sum(1 for p in parsed if p is not None)}

    def s11_evaluate(obs, ctx):
        if isinstance(obs, BaseException):
            return False, f"fuzzing raised {type(obs).__name__}: {obs}"
        ok = not obs["crashes"] and obs["responses"] == obs["calls"] and obs["rows"] == 0
        detail = f"{obs['calls']} hostile inputs handled with no crash and no database write"
        return ok, detail if ok else detail + f"; crashes: {obs['crashes'][:3]}"

    experiments.append(Experiment(
        id="S11", name="Oversized and malformed inputs to tools",
        hypothesis="Fuzzed paths, queries and evidence are rejected cleanly: no unhandled exception, no file outside the repository read, nothing written.",
        fault=FaultSpec("fuzz_input", {"kinds": ["oversized", "malformed_json", "awkward_strings"]}),
        blast_radius=pipeline_radius, targets=[PIPELINE_TARGET, DB_TARGET],
        steady_state=[reset, db_probe()], action=s11_action, evaluate=s11_evaluate,
    ))

    return experiments


def run_self_suite(seed: int = 0, timeout_seconds: float = 120.0, only: list[str] | None = None) -> list[ExperimentResult]:
    """Run the self-resilience experiments and return their results."""
    env = _Environment(seed)
    env.install()
    try:
        experiments = build_experiments(env)
        if only:
            wanted = {item.upper() for item in only}
            experiments = [e for e in experiments if e.id in wanted]
        runner = ChaosRunner(seed=seed, global_timeout_seconds=timeout_seconds)
        return runner.run_all(experiments)
    finally:
        env.restore()


def print_results(results: list[ExperimentResult], seed: int) -> None:
    print(f"RepoRover self-resilience chaos suite (seed {seed})")
    print("-" * 78)
    for result in results:
        print(f"[{result.status.upper():^7}] {result.experiment_id}  {result.name}")
        print(f"          hypothesis: {result.hypothesis}")
        print(f"          result:     {result.detail}")
        if result.recovered is not None:
            print(f"          recovered after rollback: {'yes' if result.recovered else 'NO'}")
    print("-" * 78)
    counts = summarize(results)
    print("Summary: " + ", ".join(f"{count} {status}" for status, count in sorted(counts.items())))
