"""`reporover demo`: the complete pipeline, offline, from recorded LLM traffic.

A cassette is a JSON file holding every model request and response from one
real run against the bundled fixture `bench/fixtures/static_ledger`.

    reporover demo            replay the cassette: no network, no API key, no database
    reporover demo --record   run against the configured model and record a new cassette

Replay runs the real pipeline: fingerprint, detector engine, the five agents
with their tools, every guardrail, the report renderer and the chaos plan
generator. Only two things are substituted: the model client returns the
recorded responses in order, and the database is an in-memory stand-in.

What a cassette contains
- Message text and model responses only. HTTP headers are never captured, so
  no API key is recorded. As a second line of defence every string is scrubbed
  of configured secret values, common key formats, the local fixture path and
  the user's home directory before it is written.
- Responses are matched to requests by agent and order, not by content, so a
  cassette replays on any machine regardless of where the checkout lives.
"""

from dotenv import load_dotenv

load_dotenv()

import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_NAME = "static_ledger"
FIXTURE_PLACEHOLDER = "<FIXTURE>"
HOME_PLACEHOLDER = "<HOME>"
SECRET_PLACEHOLDER = "<REDACTED>"
CASSETTE_VERSION = 1
AGENT_NAMES = ("Scout", "Inspector", "Red Teamer", "Optimizer", "Synthesizer")

_SECRET_ENV_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "DATABASE_URL")
_KEY_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"\bgsk_[0-9A-Za-z]{20,}"),
    re.compile(r"\bsk-[0-9A-Za-z_\-]{20,}"),
    re.compile(r"(?i)bearer\s+[0-9A-Za-z_\-.=]{16,}"),
    re.compile(r"postgres(?:ql)?://[^\s\"']+"),
)


def fixture_path() -> Path:
    return PROJECT_ROOT / "bench" / "fixtures" / FIXTURE_NAME


def cassette_path() -> Path:
    override = os.getenv("REPOROVER_DEMO_CASSETTE")
    return Path(override) if override else PROJECT_ROOT / "demo" / "cassettes" / f"{FIXTURE_NAME}.json"


class CassetteError(RuntimeError):
    """The cassette is missing, malformed, or does not match the pipeline."""


# --------------------------------------------------------------------------- #
# Scrubbing
# --------------------------------------------------------------------------- #
class Scrubber:
    """Removes secrets and machine-specific paths from recorded text."""

    def __init__(self, fixture: Path, environ: dict[str, str] | None = None):
        environ = dict(os.environ if environ is None else environ)
        self.secrets = sorted(
            {value for name, value in environ.items()
             if any(hint in name.upper() for hint in _SECRET_ENV_HINTS) and len(value) >= 8},
            key=len, reverse=True,
        )
        self.paths = self._variants(str(fixture), FIXTURE_PLACEHOLDER) + self._variants(str(Path.home()), HOME_PLACEHOLDER)
        self.redactions = 0

    @staticmethod
    def _variants(path: str, placeholder: str) -> list[tuple[str, str]]:
        forms = {path, path.replace("\\", "/"), path.replace("\\", "\\\\"), path.replace("/", "\\")}
        return [(form, placeholder) for form in sorted(forms, key=len, reverse=True) if len(form) > 3]

    def __call__(self, text: Any) -> str:
        text = str(text or "")
        for secret in self.secrets:
            if secret in text:
                self.redactions += text.count(secret)
                text = text.replace(secret, SECRET_PLACEHOLDER)
        for pattern in _KEY_PATTERNS:
            text, count = pattern.subn(SECRET_PLACEHOLDER, text)
            self.redactions += count
        for needle, placeholder in self.paths:
            text = re.sub(re.escape(needle), lambda _m: placeholder, text, flags=re.IGNORECASE)
        return text


def find_secrets(text: str, environ: dict[str, str] | None = None) -> list[str]:
    """Names of anything secret-looking still present in text (for verification)."""
    environ = dict(os.environ if environ is None else environ)
    found = [
        name for name, value in environ.items()
        if any(hint in name.upper() for hint in _SECRET_ENV_HINTS) and len(value) >= 8 and value in text
    ]
    found += [pattern.pattern for pattern in _KEY_PATTERNS if pattern.search(text)]
    return found


# --------------------------------------------------------------------------- #
# Recording and replaying model clients
# --------------------------------------------------------------------------- #
def agent_of(messages: list[Any]) -> str:
    system = str(getattr(messages[0], "content", "")) if messages else ""
    for name in AGENT_NAMES:
        if f"You are {name} -" in system:
            return name
    return "unknown"


class RecordingChat:
    """Wraps a real chat model and appends each exchange to the cassette."""

    def __init__(self, chat: Any, interactions: list[dict[str, Any]], scrub: Scrubber):
        self._chat = chat
        self._interactions = interactions
        self._scrub = scrub
        self.model_name = chat.model_name
        self.temperature = chat.temperature

    def invoke(self, messages, stop=None):
        response = self._chat.invoke(messages, stop=stop)
        content = response.content if isinstance(response.content, str) else json.dumps(response.content)
        usage = getattr(response, "usage_metadata", None) or {}
        self._interactions.append(
            {
                "agent": agent_of(messages),
                "request": {
                    "messages": [
                        {"role": type(m).__name__.replace("Message", "").lower(), "content": self._scrub(m.content)}
                        for m in messages
                    ],
                    "stop": stop,
                },
                "response": {
                    "content": self._scrub(content),
                    "usage": {k: usage.get(k, 0) for k in ("input_tokens", "output_tokens", "total_tokens")},
                },
            }
        )
        return response


class ReplayChat:
    """Returns the recorded responses for each agent, in order. No network."""

    def __init__(self, cassette: dict[str, Any], fixture: Path, counters: dict[str, int]):
        self.model_name = cassette["meta"].get("model", "recorded-model")
        self.temperature = 0.0
        # Forward slashes: recorded answers often embed the path inside JSON,
        # where a restored Windows backslash would be an invalid escape.
        self._fixture = fixture.as_posix()
        self._counters = counters
        self._by_agent: dict[str, list[dict[str, Any]]] = {}
        for interaction in cassette["interactions"]:
            self._by_agent.setdefault(interaction["agent"], []).append(interaction)

    def invoke(self, messages, stop=None):
        agent = agent_of(messages)
        index = self._counters.get(agent, 0)
        recorded = self._by_agent.get(agent, [])
        if index >= len(recorded):
            raise CassetteError(
                f"The cassette has no response #{index + 1} for the {agent} agent. The pipeline has changed "
                "since it was recorded; record a new one with `reporover demo --record`."
            )
        self._counters[agent] = index + 1
        self._counters["total"] = self._counters.get("total", 0) + 1
        content = recorded[index]["response"]["content"].replace(FIXTURE_PLACEHOLDER, self._fixture)
        return SimpleNamespace(content=content, usage_metadata=recorded[index]["response"].get("usage"))


def load_cassette(path: Path | None = None) -> dict[str, Any]:
    path = path or cassette_path()
    try:
        cassette = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        raise CassetteError(
            f"No cassette at {path}. Record one with `reporover demo --record` (needs the model API key once)."
        ) from None
    except json.JSONDecodeError as exc:
        raise CassetteError(f"The cassette at {path} is not valid JSON: {exc}") from None
    if cassette.get("version") != CASSETTE_VERSION or not isinstance(cassette.get("interactions"), list):
        raise CassetteError(f"The cassette at {path} has an unsupported format.")
    return cassette


# --------------------------------------------------------------------------- #
# Independent re-verification of the final findings
# --------------------------------------------------------------------------- #
def _normalized(text: str) -> str:
    return "\n".join(line.strip() for line in str(text).replace("\r\n", "\n").split("\n") if line.strip())


def reverify_finding(root: Path, finding: dict[str, Any], detector_ids: set[str]) -> bool:
    """Re-check a finding's evidence against the files on disk, without the ledger."""
    evidence = finding.get("evidence") or []
    if not evidence:
        return False
    for item in evidence:
        if item.get("kind") == "detector":
            if item.get("id") not in detector_ids:
                return False
        elif item.get("kind") == "quote":
            try:
                source = _normalized((root / item["file"]).read_text(encoding="utf-8", errors="replace"))
            except (OSError, KeyError):
                return False
            if _normalized(item.get("quote", "")) not in source:
                return False
        elif item.get("file") and item.get("line"):
            try:
                lines = (root / item["file"]).read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                return False
            first = str(item.get("snippet") or "").split("\n")[0].strip()
            if item["line"] > len(lines) or not first or not lines[item["line"] - 1].strip().startswith(first):
                return False
    return True


# --------------------------------------------------------------------------- #
# The demo run
# --------------------------------------------------------------------------- #
class _Output:
    def __init__(self, quiet: bool, pace: float):
        self.quiet = quiet
        self.pace = pace

    def say(self, text: str = "") -> None:
        if not self.quiet:
            print(text, flush=True)

    def stage(self, number: int, title: str) -> None:
        if not self.quiet:
            if self.pace:
                time.sleep(self.pace)
            print(f"\n[{number}/8] {title}", flush=True)


def run_demo(
    record: bool = False,
    quiet: bool = False,
    write_outputs: bool = True,
    pace: float = 0.0,
    verbose: bool = False,
    output_directory: Path | None = None,
) -> dict[str, Any]:
    """Run the pipeline on the bundled fixture, replaying or recording a cassette.

    Returns {"report", "metrics", "run_directory", "chaos_directory", "cassette"}.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    from repo_rover import analysis_crew, catalog, db_tools, resilience
    from repo_rover.chaos import generator
    from repo_rover.chaos.self_test import FakeDatabase
    from repo_rover.read_ledger import get_active_run, set_active_run

    out = _Output(quiet, pace)
    fixture = fixture_path()
    if not fixture.is_dir():
        raise CassetteError(f"The bundled fixture is missing: {fixture}")
    cassette = None if record else load_cassette()
    launch = Path.cwd()
    base_output = Path(output_directory) if output_directory else launch
    scratch = Path(tempfile.mkdtemp(prefix="reporover_demo_"))

    saved_env = {name: os.environ.get(name) for name in ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL", "DATABASE_URL", "LLM_FALLBACK_MODELS")}
    saved = {"psycopg2": db_tools.psycopg2, "sleep": resilience.SLEEP, "loader": analysis_crew.load_catalog_entries,
             "run": get_active_run()}
    database = FakeDatabase()
    interactions: list[dict[str, Any]] = []
    counters: dict[str, int] = {}

    try:
        if record:
            for name in ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL"):
                if not os.environ.get(name):
                    raise CassetteError(f"Recording needs {name} in the environment or .env.")
        else:
            os.environ.update({"LLM_API_KEY": "replay-no-key", "LLM_MODEL": cassette["meta"].get("model", "recorded-model"),
                               "LLM_BASE_URL": "http://127.0.0.1:9/replay", "LLM_FALLBACK_MODELS": ""})
            resilience.SLEEP = lambda _seconds: None
        # The demo never touches a real database, in either mode.
        os.environ["DATABASE_URL"] = "postgresql://demo:demo@in-process/fake"
        db_tools.psycopg2 = SimpleNamespace(connect=database.connect)
        entries = catalog.load_catalog(catalog.catalog_path())["entries"]
        analysis_crew.load_catalog_entries = lambda prefer_db=True: (entries, "yaml")

        mode = "RECORDING live model traffic" if record else "REPLAY of recorded traffic - offline"
        out.say("=" * 78)
        out.say(f"RepoRover demo  |  fixture: bench/fixtures/{FIXTURE_NAME}  |  {mode}")
        out.say("=" * 78)

        crew = analysis_crew.run_analyzer(str(fixture), checkpoint_directory=scratch)
        run = get_active_run()

        out.stage(1, "Deterministic fingerprint (no LLM)")
        out.say(run.fingerprint.summary_text())
        out.stage(2, f"Detector engine: {len(run.detector_findings)} catalogue rules fired (no LLM)")
        for finding in list(run.detector_findings.values())[:8]:
            hit = finding["hits"][0]
            where = f"{hit['file']}:{hit['line']}" if hit.get("file") and hit.get("line") else "(whole repo)"
            out.say(f"  {finding['id']:<4} {finding['slug']:<40} {where}")
        if len(run.detector_findings) > 8:
            out.say(f"  ... and {len(run.detector_findings) - 8} more")

        scrub = Scrubber(fixture)
        seen: set[int] = set()
        for agent in crew.agents:
            agent.verbose = verbose
            if id(agent.llm) in seen:
                continue
            seen.add(id(agent.llm))
            if record:
                agent.llm._chain = [RecordingChat(chat, interactions, scrub) for chat in agent.llm._chain]
            else:
                agent.llm._chain = [ReplayChat(cassette, fixture, counters)]
        crew.verbose = verbose

        stage_titles = {
            "Scout": "Scout: business intent, backed by quotes from the read ledger",
            "Inspector": "Inspector: judging the deterministic hits",
            "Red Teamer": "Red Teamer: scenarios tied to the fingerprint and to evidence",
            "Optimizer": "Optimizer: complexity findings with verified quotes",
            "Synthesizer": "Synthesizer: wording only; labels and layout are assigned by code",
        }
        progress = {"n": 2}

        def on_task(task_output) -> None:
            name = next((n for n in AGENT_NAMES if str(getattr(task_output, "agent", "")).startswith(n)), "Agent")
            progress["n"] += 1
            out.stage(progress["n"], stage_titles.get(name, name))
            labels = {label: sum(1 for f in run.findings.values() if f["label"] == label)
                      for label in ("Verified", "Detector-Hit", "Hypothesis")}
            rejected = len(run.artifacts.get("guardrail_rejections", []))
            out.say(f"  read ledger: {len(run.entries)} file(s), {run.lines_served} line(s) served to agents")
            out.say(f"  findings so far: {labels['Verified']} Verified, {labels['Detector-Hit']} Detector-Hit, "
                    f"{labels['Hypothesis']} Hypothesis  |  outputs rejected by guardrails: {rejected}")

        crew.task_callback = on_task
        started = time.monotonic()
        result = crew.kickoff()
        elapsed = time.monotonic() - started
        os.chdir(launch)
        report = str(result)

        detector_ids = set(run.detector_findings)
        backed = [f for f in run.findings.values() if f["label"] != "Hypothesis"]
        meta = cassette["meta"] if cassette else {}
        metrics = {
            "fixture": FIXTURE_NAME,
            "model": meta.get("model") or os.environ.get("LLM_MODEL", ""),
            "recorded_at": meta.get("recorded_at") or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "llm_calls": len(interactions) if record else counters.get("total", 0),
            "guardrail_rejections": len(run.artifacts.get("guardrail_rejections", [])),
            "status": run.final_status(),
            "degraded_reasons": list(run.degraded_reasons),
            "verified": sum(1 for f in run.findings.values() if f["label"] == "Verified"),
            "detector_hits": sum(1 for f in run.findings.values() if f["label"] == "Detector-Hit"),
            "hypotheses": sum(1 for f in run.findings.values() if f["label"] == "Hypothesis"),
            "unverified_backed_findings": sum(1 for f in backed if not reverify_finding(fixture, f, detector_ids)),
            "discoveries_inserted": len(run.inserted_discoveries),
            "seconds": round(elapsed, 1),
        }
        # Fidelity: a replay must end in the same state as the run it recorded.
        metrics["matches_recording"] = record or (
            metrics["status"] == meta.get("run_status") and metrics["llm_calls"] == meta.get("interactions")
        )

        saved_cassette = None
        if record:
            saved_cassette = cassette_path()
            saved_cassette.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": CASSETTE_VERSION,
                "meta": {
                    "fixture": FIXTURE_NAME,
                    "model": os.environ.get("LLM_MODEL", ""),
                    "provider_host": urlsplit(os.environ.get("LLM_BASE_URL", "")).hostname or "",
                    "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "interactions": len(interactions),
                    "run_status": metrics["status"],
                    "note": "Message text only. No HTTP headers or API keys are stored; paths and secrets are scrubbed.",
                },
                "interactions": interactions,
            }
            text = json.dumps(payload, indent=1, ensure_ascii=False)
            leaks = find_secrets(text)
            if leaks:
                raise CassetteError(f"Refusing to save the cassette: secret-like content remains ({leaks}).")
            saved_cassette.write_text(text, encoding="utf-8")

        run_directory = chaos_directory = None
        if write_outputs:
            run_directory = base_output / analysis_crew.RUNS_DIRECTORY / f"demo_{datetime.now():%Y%m%d_%H%M%S}"
            run.save(run_directory)
            (run_directory / "prompt.md").write_text(report, encoding="utf-8")
            chaos_directory, plan = generator.generate_plan(
                list(run.findings.values()), run.fingerprint.stack_trigger, str(fixture), base_output / "out")
            out.stage(progress["n"] + 1, f"Chaos plan generated from evidence-backed findings ({len(plan['experiments'])} experiments, not executed)")
            for experiment in plan["experiments"]:
                cited = ", ".join(f["id"] for f in experiment["findings"])
                out.say(f"  {experiment['id']}  {experiment['tool']:<10} {experiment['script']:<42} <- {cited}")

        out.say("\n" + "-" * 78)
        out.say(f"Run status: {metrics['status']}  |  LLM calls: {metrics['llm_calls']}  |  "
                f"guardrail rejections: {metrics['guardrail_rejections']}  |  {metrics['seconds']}s")
        out.say(f"Report findings: {metrics['verified']} [Verified], {metrics['detector_hits']} [Detector-Hit], "
                f"{metrics['hypotheses']} [Hypothesis]")
        out.say(f"Evidence-backed findings failing independent re-verification: {metrics['unverified_backed_findings']}")
        if not record:
            out.say("Replay matches the recorded run: " + ("yes" if metrics["matches_recording"] else
                    f"NO (recorded status {meta.get('run_status')}, {meta.get('interactions')} calls)"))
        if run_directory:
            out.say(f"Report:      {run_directory / 'prompt.md'}")
            out.say(f"Read ledger: {run_directory / 'ledger.json'}")
            out.say(f"Chaos plan:  {chaos_directory}")
        if saved_cassette:
            out.say(f"Cassette recorded: {saved_cassette} ({len(interactions)} interactions, {scrub.redactions} redactions)")
        return {"report": report, "metrics": metrics, "run_directory": run_directory,
                "chaos_directory": chaos_directory, "cassette": saved_cassette}
    finally:
        os.chdir(launch)
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        db_tools.psycopg2 = saved["psycopg2"]
        resilience.SLEEP = saved["sleep"]
        analysis_crew.load_catalog_entries = saved["loader"]
        set_active_run(saved["run"])
        import shutil

        shutil.rmtree(scratch, ignore_errors=True)
