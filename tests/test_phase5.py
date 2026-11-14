"""Phase 5 offline tests: the benchmark and the offline demo.

    uv run python -m unittest discover -s tests -v
"""

import importlib.util
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import repo_rover
from repo_rover import catalog, cli, demo_mode

PROJECT_ROOT = Path(repo_rover.__file__).resolve().parents[2]


def load_benchmark():
    spec = importlib.util.spec_from_file_location("reporover_benchmark", PROJECT_ROOT / "bench" / "benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = load_benchmark()
        with mock.patch.object(cls.bench, "agent_stage_metrics", return_value=None):
            cls.results = cls.bench.run_benchmark()

    def test_metric_arithmetic(self):
        m = self.bench.prf(8, 2, 2)
        self.assertEqual((m["precision"], m["recall"]), (0.8, 0.8))
        self.assertAlmostEqual(m["f1"], 0.8)
        self.assertEqual(self.bench.prf(0, 0, 0)["f1"], 0.0)
        self.assertAlmostEqual(self.bench.prf(3, 1, 0)["f1"], 2 * 0.75 * 1.0 / 1.75)

    def test_three_fixtures_with_valid_ground_truth(self):
        fixtures = self.bench.load_fixtures()
        self.assertEqual([name for name, _, _ in fixtures], ["express_api", "flask_orders", "static_ledger"])
        known = {entry["slug"] for entry in catalog.load_catalog(catalog.DEFAULT_CATALOG_PATH)["entries"]}
        for name, root, expected in fixtures:
            slugs = [item["slug"] for item in expected["present"]]
            self.assertEqual(len(slugs), len(set(slugs)), name)
            self.assertTrue(set(slugs) <= known, name)
            self.assertFalse(set(slugs) & set(expected["judged_not_present"]), name)
            self.assertTrue(all(item["where"] and item["kind"] in ("seeded", "incidental") for item in expected["present"]))
            source_lines = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in root.rglob("*")
                               if p.suffix in (".py", ".js", ".html"))
            self.assertLess(source_lines, 120, f"{name} must stay tiny")

    def test_totals_are_consistent_with_per_fixture_counts(self):
        overall = self.results["overall"]
        for key in ("tp", "fp", "fn"):
            self.assertEqual(overall[key], sum(f["all"][key] for f in self.results["fixtures"]))
        self.assertEqual(overall["tp"] + overall["fn"], sum(f["labelled"] for f in self.results["fixtures"]))
        for fixture in self.results["fixtures"]:
            self.assertEqual(len(fixture["false_positives"]), fixture["all"]["fp"])
            self.assertEqual(len(fixture["false_negatives"]), fixture["all"]["fn"])

    def test_benchmark_is_not_trivially_perfect(self):
        """The fixtures contain flaws the detectors miss and hits judged wrong."""
        overall = self.results["overall"]
        self.assertGreater(overall["fp"], 0)
        self.assertGreater(overall["fn"], 0)
        self.assertGreater(overall["precision"], 0.5)
        self.assertGreater(overall["recall"], 0.5)

    def test_every_cited_location_is_reverified(self):
        hallucination = self.results["hallucination"]
        self.assertGreater(hallucination["located_hits"], 30)
        self.assertEqual(hallucination["unverified"], 0)
        self.assertEqual(hallucination["rate"], 0.0)

    def test_reverification_detects_a_wrong_citation(self):
        root = PROJECT_ROOT / "bench" / "fixtures" / "flask_orders"
        good = {"file": "app.py", "line": 61, "end_line": 61, "snippet": 'app.run(host="0.0.0.0", debug=True)'}
        self.assertTrue(self.bench.verify_hit(root, good))
        self.assertFalse(self.bench.verify_hit(root, {**good, "line": 12, "end_line": 12}))
        self.assertFalse(self.bench.verify_hit(root, {**good, "snippet": "os.system(user_input)"}))
        self.assertFalse(self.bench.verify_hit(root, {**good, "file": "missing.py"}))
        self.assertFalse(self.bench.verify_hit(root, {**good, "line": 9999, "end_line": 9999}))
        self.assertIsNone(self.bench.verify_hit(root, {"file": None, "line": None, "snippet": ""}))

    def test_results_document_reports_the_measured_numbers(self):
        text = self.bench.render_results(self.results)
        overall = self.results["overall"]
        self.assertIn(f"| All rules | {overall['tp']} | {overall['fp']} | {overall['fn']} |", text)
        self.assertIn(self.bench.pct(overall["precision"]), text)
        self.assertIn("## Technical Strength", text)
        summary = self.bench.technical_strength(self.results)
        self.assertEqual(summary.count(". ") + 1, 3, "the summary must be three sentences")
        self.assertIn("not how it performs on arbitrary real-world repositories", summary)
        for fixture in self.results["fixtures"]:
            for slug in fixture["false_positives"] + fixture["false_negatives"]:
                self.assertIn(f"`{slug}`", text)

    def test_committed_results_file_matches_a_fresh_run(self):
        committed = (PROJECT_ROOT / "bench" / "RESULTS.md").read_text(encoding="utf-8")
        overall = self.results["overall"]
        self.assertIn(f"| All rules | {overall['tp']} | {overall['fp']} | {overall['fn']} |", committed)
        self.assertIn(f"{overall['f1']:.3f}", committed)

    def test_cli_bench_command(self):
        with mock.patch("builtins.print"), self.assertRaises(SystemExit) as caught:
            cli.main(["bench", "--no-write"])
        self.assertEqual(caught.exception.code, 0)


class TestCassetteSafety(unittest.TestCase):
    def test_scrubber_removes_secrets_and_paths(self):
        fixture = Path(tempfile.gettempdir()) / "checkout" / "bench" / "fixtures" / "static_ledger"
        env = {"LLM_API_KEY": "live-key-abcdef123456", "DATABASE_URL": "postgresql://u:pw@host/db", "PATH": "/usr/bin"}
        scrub = demo_mode.Scrubber(fixture, env)
        text = (
            f"key live-key-abcdef123456 db postgresql://u:pw@host/db "
            f"path {fixture}\\index.html json {str(fixture).replace(chr(92), chr(92) * 2)} "
            f"posix {fixture.as_posix()}/index.html google AIza{'B' * 35} bearer Bearer abcdefghijklmnop1234"
        )
        cleaned = scrub(text)
        for secret in ("live-key-abcdef123456", "pw@host", "AIzaBBBB", "abcdefghijklmnop1234", "checkout"):
            self.assertNotIn(secret, cleaned)
        self.assertEqual(cleaned.count(demo_mode.FIXTURE_PLACEHOLDER), 3)
        self.assertIn("/usr/bin", scrub("PATH is /usr/bin"))  # non-secret values are untouched
        self.assertGreaterEqual(scrub.redactions, 4)
        self.assertEqual(demo_mode.find_secrets(cleaned, env), [])
        self.assertTrue(demo_mode.find_secrets(text, env))

    def test_replay_returns_responses_per_agent_in_order(self):
        fixture = Path("C:/work/fixture")
        cassette = {"meta": {"model": "m"}, "interactions": [
            {"agent": "Scout", "response": {"content": "first"}},
            {"agent": "Inspector", "response": {"content": "judge"}},
            {"agent": "Scout", "response": {"content": f"second {demo_mode.FIXTURE_PLACEHOLDER}/a.py"}},
        ]}
        counters = {}
        chat = demo_mode.ReplayChat(cassette, fixture, counters)
        scout = [SimpleNamespace(content="You are Scout - Repository Context Mapper. ...")]
        inspector = [SimpleNamespace(content="You are Inspector - Known Failure Judge. ...")]
        self.assertEqual(chat.invoke(scout).content, "first")
        self.assertEqual(chat.invoke(inspector).content, "judge")
        self.assertEqual(chat.invoke(scout).content, "second C:/work/fixture/a.py")
        self.assertEqual(counters["total"], 3)
        with self.assertRaises(demo_mode.CassetteError) as caught:
            chat.invoke(scout)
        self.assertIn("reporover demo --record", str(caught.exception))

    def test_missing_or_corrupt_cassette_is_reported_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(demo_mode.CassetteError):
                demo_mode.load_cassette(Path(directory) / "none.json")
            bad = Path(directory) / "bad.json"
            bad.write_text("{", encoding="utf-8")
            with self.assertRaises(demo_mode.CassetteError):
                demo_mode.load_cassette(bad)
            bad.write_text('{"version": 99, "interactions": []}', encoding="utf-8")
            with self.assertRaises(demo_mode.CassetteError):
                demo_mode.load_cassette(bad)

    def test_telemetry_and_trace_upload_are_off_by_default(self):
        self.assertEqual(os.environ.get("CREWAI_DISABLE_TELEMETRY"), "true")
        self.assertEqual(os.environ.get("OTEL_SDK_DISABLED"), "true")
        self.assertEqual(os.environ.get("CREWAI_TRACING_ENABLED"), "false")


@unittest.skipUnless(demo_mode.cassette_path().is_file(), "no cassette recorded; run `reporover demo --record`")
class TestOfflineDemo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cassette_text = demo_mode.cassette_path().read_text(encoding="utf-8")
        cls.cassette = json.loads(cls.cassette_text)

        cls.outbound = []
        real_connect = socket.socket.connect

        def loopback_only(sock, address, *args):
            # Windows' event loop wires itself up over a loopback socket pair;
            # that never leaves the machine. Anything else is egress.
            host = address[0] if isinstance(address, tuple) else str(address)
            if host not in ("127.0.0.1", "::1", "localhost"):
                cls.outbound.append(address)
                raise AssertionError(f"the offline demo attempted a network connection to {address}")
            return real_connect(sock, address, *args)

        def no_network(*args, **kwargs):
            cls.outbound.append(args[:1])
            raise AssertionError("the offline demo attempted a network connection")

        cls.workdir = tempfile.TemporaryDirectory()
        with mock.patch.object(socket.socket, "connect", loopback_only), \
             mock.patch.object(socket, "create_connection", no_network), \
             mock.patch.object(socket, "getaddrinfo", no_network):
            cls.outcome = demo_mode.run_demo(quiet=True, output_directory=Path(cls.workdir.name))
            cls.second = demo_mode.run_demo(quiet=True, write_outputs=False)

    @classmethod
    def tearDownClass(cls):
        cls.workdir.cleanup()

    def test_cassette_holds_no_secrets_or_personal_paths(self):
        self.assertEqual(demo_mode.find_secrets(self.cassette_text), [])
        self.assertNotIn(str(Path.home()), self.cassette_text)
        self.assertNotIn(Path.home().name, self.cassette_text)
        self.assertNotIn("authorization", self.cassette_text.lower().replace("unauthorized", ""))
        meta = self.cassette["meta"]
        self.assertEqual(meta["fixture"], "static_ledger")
        self.assertEqual(meta["interactions"], len(self.cassette["interactions"]))
        self.assertTrue(meta["model"])

    def test_replay_completes_without_network_or_credentials(self):
        metrics = self.outcome["metrics"]
        self.assertEqual(self.outbound, [], "no outbound connection or DNS lookup may be attempted")
        self.assertTrue(metrics["matches_recording"])
        self.assertEqual(metrics["status"], self.cassette["meta"]["run_status"])
        self.assertEqual(metrics["llm_calls"], len(self.cassette["interactions"]))
        self.assertEqual(metrics["unverified_backed_findings"], 0)
        self.assertGreater(metrics["detector_hits"], 0)

    def test_report_is_labelled_and_written(self):
        report = self.outcome["report"]
        self.assertIn("# Work Items", report)
        self.assertTrue(report.rstrip().split("# Hypotheses (Unverified)")[-1])
        for line in (l for l in report.splitlines() if l[:1].isdigit() and ". [" in l):
            self.assertRegex(line, r"^\d+\. \[(Verified|Detector-Hit|Hypothesis)\] ")
        self.assertTrue((self.outcome["run_directory"] / "prompt.md").is_file())
        self.assertTrue((self.outcome["run_directory"] / "ledger.json").is_file())

    def test_playwright_chaos_scripts_are_generated(self):
        chaos = self.outcome["chaos_directory"]
        plan = json.loads((chaos / "plan.json").read_text(encoding="utf-8"))
        scripts = {experiment["script"] for experiment in plan["experiments"]}
        self.assertIn("playwright/storage-quota.spec.js", scripts)
        for script in scripts:
            self.assertIn("Refusing to run: BASE_URL", (chaos / script).read_text(encoding="utf-8"))

    def test_replay_is_deterministic(self):
        self.assertEqual(self.outcome["report"], self.second["report"])
        keys = ("status", "llm_calls", "guardrail_rejections", "verified", "detector_hits", "hypotheses")
        self.assertEqual([self.outcome["metrics"][k] for k in keys], [self.second["metrics"][k] for k in keys])

    def test_replay_leaves_process_state_untouched(self):
        import psycopg2

        from repo_rover import db_tools, resilience
        import time

        self.assertIs(db_tools.psycopg2, psycopg2)
        self.assertIs(resilience.SLEEP, time.sleep)
        self.assertNotEqual(os.environ.get("LLM_API_KEY"), "replay-no-key")


if __name__ == "__main__":
    unittest.main(verbosity=2)
