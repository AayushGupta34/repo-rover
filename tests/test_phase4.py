"""Phase 4 offline tests: chaos framework, self-resilience, target plan
generation, the execution gate and the discovery feedback loop.

    uv run python -m unittest discover -s tests -v
"""

import json
import os
import random
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import psycopg2

from repo_rover import cli, db_tools, resilience, runtime
from repo_rover.chaos import generator, self_test, target
from repo_rover.chaos.framework import (
    STATUS_ABORTED,
    STATUS_ERROR,
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_REFUSED,
    STATUS_SKIPPED,
    AbortCondition,
    BlastRadius,
    ChaosRunner,
    DataMonkey,
    Experiment,
    FaultSpec,
    NetworkMonkey,
    Probe,
    ResourceMonkey,
    is_local_target,
)
from repo_rover.checkpoint import RunCheckpoint
from repo_rover.evidence_report import InspectorGuardrail
from repo_rover.read_ledger import set_active_run
from repo_rover.resilience import (
    RetryExhausted,
    SimulatedHTTPError,
    full_jitter_delay,
    is_transient_db_error,
    is_transient_llm_error,
    retry_with_backoff,
)
from test_phase3 import PipelineFixture


class NoSleep(unittest.TestCase):
    """Record backoff sleeps instead of sleeping."""

    def setUp(self):
        self.sleeps = []
        patcher = mock.patch.object(resilience, "SLEEP", self.sleeps.append)
        patcher.start()
        self.addCleanup(patcher.stop)
        rng = mock.patch.object(resilience, "RNG", random.Random(3))
        rng.start()
        self.addCleanup(rng.stop)


class TestBackoff(NoSleep):
    def test_full_jitter_stays_within_the_exponential_ceiling(self):
        rng = random.Random(1)
        for attempt in range(8):
            ceiling = min(20.0, 0.5 * 2 ** attempt)
            samples = [full_jitter_delay(attempt, 0.5, 20.0, rng) for _ in range(200)]
            self.assertTrue(all(0.0 <= s <= ceiling for s in samples))
            self.assertGreater(max(samples), ceiling * 0.7)  # uses the whole range ("full" jitter)
            self.assertLess(min(samples), ceiling * 0.3)

    def test_retries_then_succeeds(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise SimulatedHTTPError(503)
            return "ok"

        self.assertEqual(retry_with_backoff(flaky, is_retryable=is_transient_llm_error), "ok")
        self.assertEqual(len(self.sleeps), 2)

    def test_non_retryable_error_is_raised_immediately(self):
        def bad():
            raise SimulatedHTTPError(401)

        with self.assertRaises(SimulatedHTTPError):
            retry_with_backoff(bad, is_retryable=is_transient_llm_error)
        self.assertEqual(self.sleeps, [])

    def test_gives_up_after_max_attempts(self):
        calls = {"n": 0}

        def down():
            calls["n"] += 1
            raise SimulatedHTTPError(429)

        with self.assertRaises(RetryExhausted) as caught:
            retry_with_backoff(down, is_retryable=is_transient_llm_error, max_attempts=4)
        self.assertEqual((calls["n"], caught.exception.attempts, len(self.sleeps)), (4, 4, 3))

    def test_retry_after_is_honoured_up_to_the_cap(self):
        calls = {"n": 0}

        def limited():
            calls["n"] += 1
            if calls["n"] == 1:
                raise SimulatedHTTPError(429, retry_after=7)
            return "ok"

        retry_with_backoff(limited, is_retryable=is_transient_llm_error, base_delay=0.1, max_delay=5)
        self.assertEqual(self.sleeps, [5])

    def test_error_classification(self):
        for status in (429, 500, 502, 503, 504):
            self.assertTrue(is_transient_llm_error(SimulatedHTTPError(status)))
        for status in (400, 401, 403, 404):
            self.assertFalse(is_transient_llm_error(SimulatedHTTPError(status)))
        self.assertTrue(is_transient_llm_error(TimeoutError()))
        self.assertTrue(is_transient_llm_error(resilience.EmptyCompletionError()))
        self.assertFalse(is_transient_llm_error(ValueError("bad prompt")))

        self.assertTrue(is_transient_db_error(psycopg2.OperationalError("connection reset")))
        self.assertTrue(is_transient_db_error(psycopg2.errors.QueryCanceled("statement timeout")))
        self.assertTrue(is_transient_db_error(psycopg2.errors.DeadlockDetected("deadlock")))
        self.assertFalse(is_transient_db_error(psycopg2.errors.UniqueViolation("duplicate")))
        self.assertFalse(is_transient_db_error(psycopg2.errors.SyntaxError("syntax")))


class TestModelFallback(NoSleep):
    def chats(self):
        return self_test.FakeChat("llama-3.3-70b-versatile", "primary"), self_test.FakeChat("llama-3.1-8b-instant", "smaller")

    def test_default_fallback_for_the_primary_groq_model(self):
        with mock.patch.dict(os.environ, {"LLM_MODEL": "llama-3.3-70b-versatile"}):
            os.environ.pop("LLM_FALLBACK_MODELS", None)
            self.assertEqual(runtime.fallback_model_names(), ["llama-3.1-8b-instant"])
        with mock.patch.dict(os.environ, {"LLM_MODEL": "a", "LLM_FALLBACK_MODELS": "b, c ,a"}):
            self.assertEqual(runtime.fallback_model_names(), ["b", "c"])
        with mock.patch.dict(os.environ, {"LLM_MODEL": "a", "LLM_FALLBACK_MODELS": ""}):
            self.assertEqual(runtime.fallback_model_names(), [])

    def test_non_transient_primary_failure_falls_back_without_retrying(self):
        primary, smaller = self.chats()
        primary.invoke = mock.Mock(side_effect=SimulatedHTTPError(404, "model not found"))
        llm = runtime.GroqChatLLM(primary, fallbacks=[smaller])
        self.assertEqual(llm.call("hi"), "smaller")
        self.assertEqual(primary.invoke.call_count, 1)
        self.assertEqual(self.sleeps, [])

    def test_empty_completion_is_retried(self):
        primary, smaller = self.chats()
        replies = iter([SimpleNamespace(content="  ", usage_metadata=None), SimpleNamespace(content="real", usage_metadata=None)])
        primary.invoke = lambda messages, stop=None: next(replies)
        self.assertEqual(runtime.GroqChatLLM(primary, fallbacks=[smaller]).call("hi"), "real")
        self.assertEqual(len(self.sleeps), 1)

    def test_all_models_down_raises_one_clear_error(self):
        primary, smaller = self.chats()
        primary.invoke = mock.Mock(side_effect=ConnectionError("offline"))
        smaller.invoke = mock.Mock(side_effect=ConnectionError("offline"))
        with self.assertRaises(runtime.LLMUnavailableError) as caught:
            runtime.GroqChatLLM(primary, fallbacks=[smaller]).call("hi")
        self.assertIn("All 2 model(s) failed", str(caught.exception))
        self.assertEqual(primary.invoke.call_count, runtime.LLM_MAX_ATTEMPTS)


class TestLocalTargetRule(unittest.TestCase):
    def test_local_targets_are_accepted(self):
        for value in ("http://localhost:3000", "localhost:8080", "http://127.0.0.1", "http://127.8.9.10:5000/x",
                      "http://[::1]:3000", "https://app.localhost", "http://host.docker.internal:3000", "in-process:llm"):
            self.assertTrue(is_local_target(value), value)

    def test_external_and_deceptive_targets_are_refused(self):
        for value in ("https://example.com", "http://localhost.evil.com", "http://evil.com/localhost",
                      "http://localhost@evil.com", "http://user:pw@localhost", "http://192.168.1.10",
                      "http://10.0.0.5", "http://172.17.0.2", "http://8.8.8.8", "ftp://localhost", "file:///etc/passwd",
                      "", "in-process:", "http://2130706433.example.com", "http://127.0.0.1.nip.io"):
            self.assertFalse(is_local_target(value), value)

    def test_docker_bridge_needs_an_explicit_opt_in(self):
        self.assertFalse(is_local_target("http://172.17.0.2:3000"))
        self.assertTrue(is_local_target("http://172.17.0.2:3000", allow_docker_bridge=True))
        self.assertFalse(is_local_target("http://192.168.1.10", allow_docker_bridge=True))


class Service:
    """A toy dependency for runner tests."""

    def __init__(self):
        self.healthy = True

    def fetch(self):
        return "data"


def experiment(service, **overrides) -> Experiment:
    defaults = dict(
        id="T1", name="toy", hypothesis="the caller copes", fault=FaultSpec("offline"),
        blast_radius=BlastRadius(["in-process:service"], max_duration_seconds=5), targets=["in-process:service"],
        steady_state=[Probe("service healthy", lambda: service.healthy)],
        monkeys=[NetworkMonkey(service, "fetch", mode="offline")],
        action=lambda ctx: _safe(service), evaluate=lambda obs, ctx: (obs == "fallback", "used the fallback"),
    )
    defaults.update(overrides)
    return Experiment(**defaults)


def _safe(service):
    try:
        return service.fetch()
    except ConnectionError:
        return "fallback"


class TestChaosRunner(unittest.TestCase):
    def setUp(self):
        self.service = Service()
        self.original = self.service.fetch

    def assert_rolled_back(self):
        self.assertEqual(self.service.fetch(), "data", "the fault must be removed after the experiment")

    def test_passing_experiment_injects_observes_and_rolls_back(self):
        result = ChaosRunner(seed=1).run(experiment(self.service))
        self.assertEqual((result.status, result.detail), (STATUS_PASSED, "used the fallback"))
        self.assertTrue(result.recovered and result.rollback_ok)
        self.assert_rolled_back()

    def test_disproved_hypothesis_is_a_failure(self):
        result = ChaosRunner().run(experiment(self.service, evaluate=lambda obs, ctx: (obs == "data", "expected live data")))
        self.assertEqual(result.status, STATUS_FAILED)
        self.assert_rolled_back()

    def test_rollback_runs_when_the_action_raises(self):
        rolled = []
        result = ChaosRunner().run(experiment(
            self.service, action=lambda ctx: self.service.fetch(), rollback=lambda: rolled.append(True),
            evaluate=lambda obs, ctx: (isinstance(obs, ConnectionError), "error surfaced")))
        self.assertEqual(result.status, STATUS_PASSED)
        self.assertEqual(rolled, [True])
        self.assert_rolled_back()

    def test_rollback_runs_when_evaluation_itself_crashes(self):
        rolled = []

        def broken(obs, ctx):
            raise RuntimeError("bug in the experiment")

        result = ChaosRunner().run(experiment(self.service, evaluate=broken, rollback=lambda: rolled.append(True)))
        self.assertEqual(result.status, STATUS_ERROR)
        self.assertEqual(rolled, [True])
        self.assert_rolled_back()

    def test_hard_timeout_aborts_and_rolls_back(self):
        started = time.monotonic()
        result = ChaosRunner().run(experiment(
            self.service, blast_radius=BlastRadius(["in-process:service"], max_duration_seconds=0.2),
            action=lambda ctx: time.sleep(3)))
        self.assertEqual(result.status, STATUS_ABORTED)
        self.assertIn("hard timeout", result.detail)
        self.assertLess(time.monotonic() - started, 2.0)
        self.assert_rolled_back()

    def test_global_timeout_skips_remaining_experiments(self):
        clock = iter([0.0] + [100.0] * 50)
        runner = ChaosRunner(global_timeout_seconds=10, clock=lambda: next(clock))
        result = runner.run(experiment(self.service))
        self.assertEqual(result.status, STATUS_SKIPPED)
        self.assert_rolled_back()

    def test_no_injection_when_steady_state_is_not_met(self):
        self.service.healthy = False
        monkey = NetworkMonkey(self.service, "fetch", mode="offline")
        result = ChaosRunner().run(experiment(self.service, monkeys=[monkey]))
        self.assertEqual(result.status, STATUS_ABORTED)
        self.assertIn("steady state", result.detail)
        self.assertEqual(monkey.calls, 0)

    def test_target_outside_blast_radius_or_not_local_is_refused(self):
        outside = ChaosRunner().run(experiment(self.service, targets=["in-process:other"]))
        self.assertEqual(outside.status, STATUS_REFUSED)
        external = ChaosRunner().run(experiment(
            self.service, targets=["https://example.com"], blast_radius=BlastRadius(["https://example.com"], 5)))
        self.assertEqual(external.status, STATUS_REFUSED)
        self.assertIn("not a local sandbox", external.detail)
        self.assertEqual(ChaosRunner().run(experiment(self.service, targets=[])).status, STATUS_REFUSED)
        self.assert_rolled_back()

    def test_abort_condition_and_failed_recovery(self):
        aborted = ChaosRunner().run(experiment(
            self.service, abort_conditions=[AbortCondition("fallback used", lambda obs: obs == "fallback")]))
        self.assertEqual(aborted.status, STATUS_ABORTED)

        def break_service(ctx):
            self.service.healthy = False
            return "fallback"

        unrecovered = ChaosRunner().run(experiment(self.service, action=break_service))
        self.assertEqual(unrecovered.status, STATUS_FAILED)
        self.assertFalse(unrecovered.recovered)
        self.assertIn("did not return to steady state", unrecovered.detail)

    def test_monkeys(self):
        ctx = SimpleNamespace(rng=random.Random(0))
        http = NetworkMonkey(self.service, "fetch", mode="http_error", status_code=429, failures=2)
        http.inject(ctx)
        for _ in range(2):
            with self.assertRaises(SimulatedHTTPError):
                self.service.fetch()
        self.assertEqual(self.service.fetch(), "data")
        http.rollback()

        waits = []
        slow = NetworkMonkey(self.service, "fetch", mode="latency", latency_seconds=2, sleep=waits.append)
        slow.inject(ctx)
        self.assertEqual(self.service.fetch(), "data")
        slow.rollback()
        self.assertTrue(1.0 <= waits[0] <= 3.0)

        quota = ResourceMonkey(self.service, "fetch", allow_calls=2)
        quota.inject(ctx)
        self.assertEqual([self.service.fetch(), self.service.fetch()], ["data", "data"])
        with self.assertRaises(SimulatedHTTPError):
            self.service.fetch()
        quota.rollback()
        self.assert_rolled_back()

    def test_data_monkey_is_reproducible_and_bounded(self):
        first, second = DataMonkey(seed=5), DataMonkey(seed=5)
        self.assertEqual(first.malformed_json_samples(), second.malformed_json_samples())
        self.assertEqual(first.oversized_text(5000), second.oversized_text(5000))
        self.assertNotEqual(DataMonkey(seed=6).oversized_text(5000), first.oversized_text(5000))
        self.assertEqual(len(first.oversized_text(123_456)), 123_456)
        for sample in first.malformed_json_samples():
            try:
                parsed = json.loads(sample)
            except (json.JSONDecodeError, RecursionError):
                continue
            self.assertFalse(isinstance(parsed, dict) and isinstance(parsed.get("judgments"), list))


class TestSelfResilienceSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = self_test.run_self_suite(seed=11)

    def test_every_experiment_passes_and_recovers(self):
        self.assertEqual(len(self.results), 11)
        for result in self.results:
            self.assertEqual(result.status, STATUS_PASSED, f"{result.experiment_id}: {result.detail}")
            self.assertTrue(result.recovered, result.experiment_id)
            self.assertTrue(result.rollback_ok, result.experiment_id)

    def test_suite_covers_the_required_faults(self):
        names = " | ".join(r.name for r in self.results)
        for required in ("429", "503", "statement timeout", "malformed JSON", "offline", "unreachable", "crashes"):
            self.assertIn(required, names)

    def test_suite_is_reproducible_for_a_seed_and_restores_global_state(self):
        again = self_test.run_self_suite(seed=11)
        self.assertEqual([r.detail for r in again], [r.detail for r in self.results])
        other = self_test.run_self_suite(seed=12, only=["S01"])
        self.assertNotEqual(other[0].detail, self.results[0].detail)
        self.assertIs(db_tools.psycopg2, psycopg2)
        self.assertIs(resilience.SLEEP, time.sleep)

    def test_cli_exit_code(self):
        with mock.patch("builtins.print"):
            with self.assertRaises(SystemExit) as caught:
                cli.main(["chaos", "self", "--seed", "3", "--only", "S06", "--only", "S08"])
        self.assertEqual(caught.exception.code, 0)


class TestIdempotentWrites(unittest.TestCase):
    def test_dedupe_key_is_stable_and_scoped_to_the_repository(self):
        key = db_tools.discovery_dedupe_key
        self.assertEqual(key("/repo", "Race", "Express", "Fix  it."), key("/repo", "race", "express ", "fix it."))
        self.assertNotEqual(key("/repo", "race", "Express", "Fix it."), key("/other", "race", "Express", "Fix it."))
        self.assertEqual(len(key(None, "a", "b", "c")), 64)

    def test_statements_use_on_conflict_do_nothing(self):
        source = Path(db_tools.__file__).read_text(encoding="utf-8")
        self.assertIn("ON CONFLICT (dedupe_key) DO NOTHING", source)
        self.assertIn("ON CONFLICT (run_uuid) DO NOTHING", source)
        schema = Path(db_tools.SCHEMA_PATH).read_text(encoding="utf-8")
        self.assertIn("ADD COLUMN IF NOT EXISTS dedupe_key", schema)
        self.assertIn("CREATE UNIQUE INDEX IF NOT EXISTS uq_dynamic_discoveries_dedupe_key", schema)
        self.assertNotIn("DROP TABLE", schema.upper())


class TestResumeFromCheckpoint(PipelineFixture):
    def test_resumed_crew_contains_only_unfinished_tasks(self):
        from repo_rover import analysis_crew

        launch = os.getcwd()
        self.addCleanup(os.chdir, launch)
        env = {"LLM_API_KEY": "test", "LLM_MODEL": "llama-3.3-70b-versatile", "LLM_BASE_URL": "http://127.0.0.1:1/v1",
               "DATABASE_URL": "postgresql://u:p@127.0.0.1:1/db"}
        patches = (mock.patch.dict(os.environ, env),
                   mock.patch.object(analysis_crew, "start_analysis_run", return_value=55),
                   mock.patch.object(analysis_crew, "load_catalog_entries", return_value=(self.entries, "yaml")))
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

        fresh = analysis_crew.run_analyzer(str(self.repo), checkpoint_directory=self.workspace)
        self.assertEqual(len(fresh.tasks), 5)
        self.assertEqual(len(fresh.agents[0].llm._chain), 2)  # primary plus the fallback model
        first_run = analysis_crew.get_active_run()
        InspectorGuardrail(first_run)(SimpleNamespace(raw=json.dumps({"judgments": []})))
        first_run.checkpoint.save_task("scout", '{"business_intent": "Hostel ledger."}', first_run)
        os.chdir(launch)

        resumed = analysis_crew.run_analyzer(str(self.repo), resume=True, checkpoint_directory=self.workspace)
        os.chdir(launch)
        run = analysis_crew.get_active_run()
        self.assertEqual(run.run_uuid, first_run.run_uuid)
        self.assertEqual(run.db_run_id, 55)
        self.assertEqual(run.artifacts["resumed_tasks"], ["inspector", "scout"])
        self.assertEqual(set(run.findings), set(first_run.findings))
        self.assertEqual([task.agent.role.split(" - ")[0] for task in resumed.tasks], ["Red Teamer", "Optimizer", "Synthesizer"])
        red_team_task = resumed.tasks[0]
        self.assertEqual(red_team_task.context, [])
        self.assertIn("VERIFIED OUTPUT OF THE EARLIER SCOUT TASK (restored from checkpoint)", red_team_task.description)
        self.assertIn("Hostel ledger.", red_team_task.description)

        run.checkpoint.mark_completed(run)
        os.chdir(launch)
        after_completion = analysis_crew.run_analyzer(str(self.repo), resume=True, checkpoint_directory=self.workspace)
        self.assertEqual(len(after_completion.tasks), 5)

    def test_corrupt_checkpoint_is_ignored(self):
        checkpoint = RunCheckpoint(self.workspace, self.repo)
        checkpoint.directory.mkdir(parents=True)
        checkpoint.path.write_text("{ not json", encoding="utf-8")
        completed, message = checkpoint.resume_into(self.ctx)
        self.assertEqual(completed, {})
        self.assertIn("No checkpoint", message)


class PlanFixture(PipelineFixture):
    def setUp(self):
        super().setUp()
        InspectorGuardrail(self.ctx)(SimpleNamespace(raw=json.dumps({"judgments": []})))
        self.ctx.register_finding({
            "id": "R1", "label": "Hypothesis", "source": "red_team", "title": "Imported backups corrupt the ledger",
            "severity": "high", "files": [], "evidence": [], "catalog_slug": "imported-data-not-validated"})
        self.out = self.workspace / "out"
        self.directory, self.plan = generator.generate_plan(
            list(self.ctx.findings.values()), self.ctx.fingerprint.stack_trigger, str(self.repo), self.out)

    def keys(self) -> set[str]:
        return {e["key"] for e in self.plan["experiments"]}


class TestChaosPlanGeneration(PlanFixture):
    def test_plan_is_written_under_out_repo_chaos(self):
        self.assertEqual(self.directory, self.out / "repo" / "chaos")
        for name in ("plan.json", "README.md", "playwright/playwright.config.js"):
            self.assertTrue((self.directory / name).is_file(), name)
        self.assertEqual(json.loads((self.directory / "plan.json").read_text(encoding="utf-8")), self.plan)

    def test_experiments_come_only_from_evidence_backed_findings(self):
        self.assertEqual(self.keys(), {"storage-quota", "multi-tab", "spike", "ramp", "soak", "oversized-body"})
        self.assertNotIn("malformed-import", self.keys())  # its only finding is a hypothesis
        self.assertEqual(self.plan["hypotheses_ignored"], 1)
        for experiment in self.plan["experiments"]:
            self.assertTrue(experiment["findings"])
            self.assertTrue(all(f["label"] in ("Verified", "Detector-Hit") for f in experiment["findings"]))
            for field in ("hypothesis", "steady_state_probes", "fault", "blast_radius", "abort_conditions", "rollback"):
                self.assertTrue(experiment[field], field)
            self.assertTrue((self.directory / experiment["script"]).is_file())

    def test_no_findings_means_no_scripts(self):
        directory, plan = generator.generate_plan([], "unknown", None, self.workspace / "empty")
        self.assertEqual(plan["experiments"], [])
        self.assertFalse((directory / "k6").exists())
        self.assertFalse((directory / "playwright").exists())

    def test_scripts_cite_findings_and_refuse_non_local_targets(self):
        for experiment in self.plan["experiments"]:
            source = (self.directory / experiment["script"]).read_text(encoding="utf-8")
            self.assertIn(experiment["id"], source)
            self.assertIn(experiment["findings"][0]["id"], source)
            self.assertIn("Refusing to run: BASE_URL", source)
            self.assertIn("http://localhost:3000", source)
            self.assertNotRegex(source, r"__[A-Z][A-Z_]*__", "an unfilled template token remains")

    def test_k6_scripts_have_stages_thresholds_and_discovered_routes(self):
        spike = (self.directory / "k6" / "spike.js").read_text(encoding="utf-8")
        self.assertIn("stages:", spike)
        self.assertIn("abortOnFail: true", spike)
        self.assertIn("rate_limited_responses", spike)
        self.assertIn('["/"]', spike)
        self.assertIn("duration: __ENV.CHAOS_SOAK_DURATION || '10m'", (self.directory / "k6" / "soak.js").read_text(encoding="utf-8"))
        self.assertIn("target: STEP * n", (self.directory / "k6" / "ramp.js").read_text(encoding="utf-8"))

    def test_route_discovery_skips_parameterised_paths(self):
        (self.repo / "api.js").write_text(
            "app.get('/health', h); app.get('/users/:id', h); app.post('/orders', h);\n", encoding="utf-8")
        routes, post_path = generator.discover_routes(self.repo)
        self.assertEqual(routes, ["/", "/health"])
        self.assertEqual(post_path, "/orders")

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_generated_javascript_parses(self):
        for experiment in self.plan["experiments"]:
            script = self.directory / experiment["script"]
            check = script
            if experiment["tool"] == "k6":  # k6 scripts are ES modules
                check = script.with_suffix(".mjs")
                shutil.copyfile(script, check)
            done = subprocess.run([shutil.which("node"), "--check", str(check)], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, f"{experiment['script']}: {done.stderr[:400]}")


class TestExecutionGateAndFeedback(PlanFixture):
    def setUp(self):
        super().setUp()
        self.executed = []
        self.addCleanup(set_active_run, None)
        set_active_run(None)

    def run_plan(self, base_url, healthy=(True,), returncode=0, **kwargs):
        health = iter(list(healthy) + [healthy[-1]] * 100)
        queued = []

        def executor(command, cwd, env, timeout):
            self.executed.append((command, env["BASE_URL"]))
            return {"returncode": returncode, "timed_out": False, "output": "summary line"}

        def queue(node, entries):
            queued.append(node)
            return {"experiment": node["id"], "action": "queued"}

        results, feedback = target.run_plan(
            self.directory, base_url, executor=executor, probe=lambda url: (next(health), "probe"),
            which=lambda name: f"/usr/bin/{name}", catalog_entries=self.entries, queue=queue, **kwargs)
        return results, feedback, queued

    def assert_refused(self, message_part, *args, **kwargs):
        with self.assertRaises(target.GateRefused) as caught:
            self.run_plan(*args, **kwargs)
        self.assertIn(message_part, str(caught.exception))
        self.assertEqual(self.executed, [], "nothing may run when the gate refuses")

    def test_default_is_plan_only(self):
        self.assert_refused("Plan generated only", "http://localhost:3000")
        self.assert_refused("Plan generated only", "http://localhost:3000", i_own_this_target=True)

    def test_run_requires_the_ownership_flag(self):
        self.assert_refused("--i-own-this-target", "http://localhost:3000", run=True)

    def test_external_urls_are_refused_even_with_both_flags(self):
        for url in ("https://example.com", "http://localhost.evil.com", "http://192.168.1.20:3000", "http://172.17.0.3"):
            self.assert_refused("Refusing to run against", url, run=True, i_own_this_target=True)
        self.assert_refused("--base-url", None, run=True, i_own_this_target=True)

    def test_local_target_runs_every_experiment(self):
        results, feedback, _ = self.run_plan("http://localhost:3000", run=True, i_own_this_target=True, seed=9)
        self.assertEqual(len(results), len(self.plan["experiments"]))
        self.assertTrue(all(r.status == STATUS_PASSED for r in results))
        self.assertEqual(feedback, [])
        self.assertTrue(all(url == "http://localhost:3000" for _, url in self.executed))
        k6_commands = [c for c, _ in self.executed if c[0].endswith("k6")]
        self.assertEqual(k6_commands[0][1], "run")

    def test_docker_bridge_target_needs_the_sandbox_flag(self):
        self.assert_refused("Refusing to run against", "http://172.17.0.3:3000", run=True, i_own_this_target=True)
        results, _, _ = self.run_plan("http://172.17.0.3:3000", run=True, i_own_this_target=True, allow_docker_bridge=True)
        self.assertTrue(all(r.status == STATUS_PASSED for r in results))

    def test_expected_failure_is_reported_but_not_queued(self):
        results, feedback, queued = self.run_plan("http://localhost:3000", returncode=99, run=True, i_own_this_target=True)
        self.assertTrue(all(r.status == STATUS_FAILED and r.recovered for r in results))
        self.assertEqual((feedback, queued), ([], []))

    def test_unexpected_failure_state_is_queued_as_a_discovery(self):
        # Healthy before the first experiment, then the target never comes back.
        results, feedback, queued = self.run_plan(
            "http://localhost:3000", healthy=(True, False), returncode=99, run=True, i_own_this_target=True)
        self.assertEqual(results[0].status, STATUS_FAILED)
        self.assertFalse(results[0].recovered)
        self.assertEqual(len(queued), 1)
        node = queued[0]
        self.assertTrue(node["vulnerability_type"].startswith("no_recovery_after_"))
        self.assertEqual(node["stack_trigger"], self.ctx.fingerprint.stack_trigger)
        self.assertIn("did not return to steady state", node["observation"])
        self.assertEqual(feedback[0]["action"], "queued")
        self.assertTrue(all(r.status == STATUS_ABORTED for r in results[1:]))  # later ones: steady state not met

    def test_missing_tool_is_an_error_not_a_crash(self):
        results, _ = target.run_plan(self.directory, "http://localhost:3000", run=True, i_own_this_target=True,
                                     probe=lambda url: (True, "ok"), which=lambda name: None,
                                     catalog_entries=self.entries)
        self.assertTrue(all(r.status == STATUS_ERROR and "not installed" in r.detail for r in results))

    def test_queue_discovery_inserts_through_the_tool_with_chaos_evidence(self):
        cursor = mock.MagicMock()
        cursor.fetchone.return_value = (88,)
        connection = mock.MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        manager = mock.MagicMock()
        manager.__enter__.return_value = connection
        node = {"id": "C01", "vulnerability_type": "no_recovery_after_load_spike", "stack_trigger": "Express, PostgreSQL",
                "binary_flags_dict": {"user_visible": True}, "confidence": 0.6,
                "fix_recommendation": "Add a restart policy and bound the connection pool so overload sheds load.",
                "observation": "After the spike ended the target kept answering HTTP 503."}
        with mock.patch.object(db_tools, "_connection", return_value=manager):
            outcome = target.queue_discovery(node, self.entries)
            forged = json.loads(db_tools.InsertDiscoveryTool.run(
                vulnerability_type="x", stack_trigger="y", binary_flags_dict={}, fix_recommendation="z", confidence=0.5,
                evidence=[{"chaos_experiment": "C77", "observation": "an agent made this up"}]))
        self.assertEqual(outcome["action"], "queued")
        self.assertEqual(outcome["response"]["evidence_status"], "Chaos-Observed")
        params = [call.args[1] for call in cursor.execute.call_args_list if "INSERT INTO dynamic_discoveries" in call.args[0]][0]
        self.assertEqual(params[0], "no_recovery_after_load_spike")
        self.assertEqual(params[5].adapted[0]["kind"], "chaos")
        self.assertIn("error", forged)

    def test_known_failure_is_not_queued_again(self):
        known = dict(self.entries[0])
        node = {"id": "C02", "vulnerability_type": known["vulnerability_type"], "stack_trigger": known["stack_trigger"],
                "binary_flags_dict": {}, "confidence": 0.6, "fix_recommendation": known["fix_recommendation"],
                "observation": "observed"}
        with mock.patch.object(db_tools, "_connection", side_effect=AssertionError("database was touched")):
            outcome = target.queue_discovery(node, self.entries)
        self.assertEqual(outcome["action"], "already_in_catalogue")
        self.assertEqual(outcome["closest"], known["slug"])

    def test_cli_plan_without_run_executes_nothing(self):
        findings_file = self.workspace / "findings.json"
        findings_file.write_text(json.dumps({
            "target_repo": str(self.repo), "fingerprint": self.ctx.fingerprint.to_dict(),
            "report_findings": list(self.ctx.findings.values())}, default=str), encoding="utf-8")
        with mock.patch.object(target, "run_command", side_effect=AssertionError("executed")), \
             mock.patch.object(target, "http_probe", side_effect=AssertionError("probed")), \
             mock.patch("builtins.print") as printed:
            with self.assertRaises(SystemExit) as plan_only:
                cli.main(["chaos", "plan", "--findings", str(findings_file), "--out", str(self.workspace / "cli-out")])
            with self.assertRaises(SystemExit) as external:
                cli.main(["chaos", "plan", "--findings", str(findings_file), "--out", str(self.workspace / "cli-out"),
                          "--run", "--base-url", "https://example.com", "--i-own-this-target"])
        self.assertEqual(plan_only.exception.code, 0)
        self.assertEqual(external.exception.code, 2)
        self.assertTrue((self.workspace / "cli-out" / "repo" / "chaos" / "plan.json").is_file())
        self.assertIn("Refusing to run against", " ".join(str(c.args[0]) for c in printed.call_args_list if c.args))


if __name__ == "__main__":
    unittest.main(verbosity=2)
