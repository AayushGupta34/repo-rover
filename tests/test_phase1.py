"""Phase 1 offline tests: tool confinement, read limits, the read ledger,
agent guardrails and graceful degradation.

No API key, network or database is needed. Run from the project root with:

    uv run python -m unittest discover -s tests -v
"""

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from repo_rover import db_tools
from repo_rover.guardrails import (
    DEGRADED_KEY,
    OptimizerGuardrail,
    ScoutGuardrail,
    extract_json_object,
    parse_line_range,
)
from repo_rover.read_ledger import RunContext, set_active_run
from repo_rover.repo_tools import (
    MAX_LINES_PER_READ,
    MAX_SEARCH_MATCHES,
    build_repo_tools,
)

APP_PY = '''import os


class Ledger:
    def __init__(self):
        self.rows = []

    def find_duplicates(self, items):
        duplicates = []
        for a in items:
            for b in items:
                if a is not b and a == b:
                    duplicates.append(a)
        return duplicates


def main():
    ledger = Ledger()
    print(ledger.find_duplicates([1, 2, 2]))
'''

SERVER_JS = '''const express = require("express");
const app = express();

function handlePayment(req, res) {
  res.json({ ok: true });
}

app.post("/pay", handlePayment);
app.listen(3000);
'''


def output(payload) -> SimpleNamespace:
    """Stand-in for a CrewAI TaskOutput: guardrails only read `.raw`."""
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(raw=raw)


class RepoFixture(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(tempfile.mkdtemp(prefix="reporover_test_"))
        self.addCleanup(shutil.rmtree, self.workspace, ignore_errors=True)
        self.addCleanup(set_active_run, None)

        self.repo = self.workspace / "repo"
        files = {
            "README.md": "# Hostel Ledger\n\nTracks rooms and payments for hostels.\n",
            "package.json": '{\n  "name": "hostel",\n  "dependencies": {"express": "^4.19.0"}\n}\n',
            "src/app.py": APP_PY,
            "src/server.js": SERVER_JS,
            "src/big.py": "".join(f"value_{i} = {i}\n" for i in range(1, 1001)),
            ".env": "API_KEY=super-secret-value\n",
            ".env.example": "API_KEY=placeholder\n",
            ".git/config": "[core]\n\tbare = false\n",
            "node_modules/leftpad/index.js": "module.exports = function handlePayment() {};\n",
            ".venv/lib/site.py": "def handlePayment(): pass\n",
            "logo.png": "not really a png",
        }
        for relative, content in files.items():
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")

        self.outside = self.workspace / "outside_secret.txt"
        self.outside.write_text("TOP SECRET outside the repo\n", encoding="utf-8")

        self.ctx = RunContext(self.repo)
        tools = build_repo_tools(self.ctx)
        self.list_tool = tools["list"]
        self.read_tool = tools["read"]
        self.search_tool = tools["search"]
        self.outline_tool = tools["outline"]


class TestConfinement(RepoFixture):
    def test_listing_skips_noise_and_secrets(self):
        listing = self.list_tool.run()
        for expected in ("README.md", "package.json", "src/app.py", "src/server.js", ".env.example"):
            self.assertIn(f"- {expected}", listing)
        for forbidden in (".git/", "node_modules", ".venv", "logo.png", "- .env\n"):
            self.assertNotIn(forbidden, listing)
        self.assertTrue(self.ctx.in_listing("src/app.py"))
        self.assertFalse(self.ctx.in_listing("node_modules/leftpad/index.js"))

    def test_listing_ignores_directory_argument(self):
        listing = self.list_tool.run(directory=str(self.workspace))
        self.assertNotIn("outside_secret.txt", listing)

    def test_read_rejects_paths_outside_repo(self):
        escapes = [
            "../outside_secret.txt",
            "src/../../outside_secret.txt",
            str(self.outside),
            "..\\outside_secret.txt",
        ]
        for path in escapes:
            with self.subTest(path=path):
                result = self.read_tool.run(file_path=path)
                self.assertTrue(result.startswith("Error:"), result)
                self.assertNotIn("TOP SECRET", result)
        self.assertEqual(self.ctx.entries, {})

    def test_read_rejects_excluded_dirs_secrets_and_binaries(self):
        for path in (
            ".git/config",
            "node_modules/leftpad/index.js",
            ".venv/lib/site.py",
            ".env",
            "logo.png",
            "src",
            "does/not/exist.py",
        ):
            with self.subTest(path=path):
                result = self.read_tool.run(file_path=path)
                self.assertTrue(result.startswith("Error:"), result)
                self.assertNotIn("super-secret-value", result)
        self.assertEqual(self.ctx.entries, {})

    def test_outline_and_search_are_confined(self):
        self.assertTrue(
            self.outline_tool.run(file_path="../outside_secret.txt").startswith("Error:")
        )
        self.assertTrue(self.outline_tool.run(file_path=".git/config").startswith("Error:"))

        hits = self.search_tool.run(query="handlePayment")
        self.assertIn("src/server.js", hits)
        self.assertNotIn("node_modules", hits)
        self.assertNotIn(".venv", hits)
        self.assertEqual(self.search_tool.run(query="super-secret-value"), "No matches for 'super-secret-value'.")
        self.assertEqual(self.search_tool.run(query="TOP SECRET"), "No matches for 'TOP SECRET'.")

    def test_symlink_escape_is_blocked(self):
        link = self.repo / "src" / "link.txt"
        try:
            os.symlink(self.outside, link)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are not permitted on this machine")
        result = self.read_tool.run(file_path="src/link.txt")
        self.assertTrue(result.startswith("Error:"), result)
        self.assertNotIn("link.txt", self.list_tool.run())
        self.assertNotIn("TOP SECRET", self.search_tool.run(query="TOP SECRET"))


class TestReadLimits(RepoFixture):
    def test_read_is_capped_per_call(self):
        result = self.read_tool.run(file_path="src/big.py", start_line=1, line_count=5000)
        self.assertIn(f"lines 1-{MAX_LINES_PER_READ} of 1000", result)
        self.assertIn(f"value_{MAX_LINES_PER_READ} = ", result)
        self.assertNotIn(f"value_{MAX_LINES_PER_READ + 1} = ", result)
        self.assertEqual(self.ctx.lines_served, MAX_LINES_PER_READ)

    def test_run_budget_is_enforced(self):
        ctx = RunContext(self.repo, line_budget=250)
        read = build_repo_tools(ctx)["read"]
        read.run(file_path="src/big.py", start_line=1)
        second = read.run(file_path="src/big.py", start_line=201)
        self.assertIn("lines 201-250 of 1000", second)
        third = read.run(file_path="src/big.py", start_line=251)
        self.assertTrue(third.startswith("Error:"), third)
        self.assertIn("budget is exhausted", third)
        self.assertEqual(ctx.lines_served, 250)
        self.assertTrue(build_repo_tools(ctx)["search"].run(query="value").startswith("Error:"))

    def test_search_results_are_capped(self):
        result = self.search_tool.run(query="value_", path_glob="src/*.py")
        self.assertLessEqual(result.count("src/big.py:"), MAX_SEARCH_MATCHES)
        self.assertLessEqual(self.ctx.lines_served, MAX_SEARCH_MATCHES)

    def test_invalid_regex_is_reported(self):
        self.assertTrue(self.search_tool.run(query="(", is_regex=True).startswith("Error:"))


class TestLedger(RepoFixture):
    def test_ledger_records_path_ranges_and_hash(self):
        self.read_tool.run(file_path="src/app.py", start_line=1, line_count=5)
        self.read_tool.run(file_path="src/app.py", start_line=8, line_count=3)
        entry = self.ctx.entry_for("src/app.py")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.ranges(), [(1, 5), (8, 10)])
        expected_hash = hashlib.sha256((self.repo / "src" / "app.py").read_bytes()).hexdigest()
        self.assertEqual(entry.sha256, expected_hash)

        summary = self.ctx.summary()
        self.assertEqual(summary["files_read"], 1)
        self.assertEqual(summary["lines_served"], 8)
        self.assertEqual(summary["files"][0]["line_ranges"], [[1, 5], [8, 10]])
        self.assertEqual(summary["files"][0]["sha256"], expected_hash)
        json.dumps(summary)  # must be JSON-serializable for the JSONB column

    def test_ledger_is_saved_to_disk(self):
        self.read_tool.run(file_path="README.md")
        ledger_path = self.ctx.save(self.workspace / "runs" / "one")
        saved = json.loads(ledger_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["files"][0]["path"], "README.md")

    def test_outline_and_search_are_ledgered(self):
        outline = self.outline_tool.run(file_path="src/app.py")
        self.assertIn("class Ledger", outline)
        self.assertIn("def find_duplicates", outline)
        self.assertIn("OutlineFileTool", self.ctx.entry_for("src/app.py").tools)

        js_outline = self.outline_tool.run(file_path="src/server.js")
        self.assertIn("function handlePayment", js_outline)

        self.search_tool.run(query="express()")
        self.assertIn("SearchRepoTool", self.ctx.entry_for("src/server.js").tools)

    def test_path_claims_are_normalized(self):
        self.read_tool.run(file_path="src/app.py")
        for claim in ("src/app.py", "./src/app.py", "src\\app.py", str(self.repo / "src" / "app.py")):
            with self.subTest(claim=claim):
                self.assertTrue(self.ctx.has_read(claim))
        self.assertFalse(self.ctx.has_read("src/server.js"))
        self.assertFalse(self.ctx.has_read("../outside_secret.txt"))

    def test_quote_matching_is_whitespace_normalized(self):
        self.read_tool.run(file_path="src/app.py")
        reindented = "for a in items:\r\n\tfor b in items:\r\n      if a is not b and a == b:"
        self.assertTrue(self.ctx.quote_matches("src/app.py", reindented)[0])
        with_line_numbers = "    9| for a in items:\n   10| for b in items:"
        self.assertTrue(self.ctx.quote_matches("src/app.py", with_line_numbers)[0])
        partial_line = "duplicates.append(a)"
        self.assertTrue(self.ctx.quote_matches("src/app.py", partial_line)[0])

        self.assertFalse(self.ctx.quote_matches("src/app.py", "for a in itemz:\nfor b in items:")[0])
        self.assertFalse(self.ctx.quote_matches("src/app.py", "for x in range(10):")[0])
        self.assertFalse(self.ctx.quote_matches("src/app.py", "a")[0])
        self.assertFalse(self.ctx.quote_matches("src/server.js", "const app = express();")[0])

    def test_quote_must_come_from_served_lines(self):
        self.read_tool.run(file_path="src/app.py", start_line=1, line_count=3)
        self.assertFalse(self.ctx.quote_matches("src/app.py", "duplicates.append(a)")[0])
        self.assertFalse(self.ctx.lines_were_served("src/app.py", 8, 12))
        self.assertTrue(self.ctx.lines_were_served("src/app.py", 1, 3))


class TestScoutGuardrail(RepoFixture):
    def setUp(self):
        super().setUp()
        self.list_tool.run()
        self.read_tool.run(file_path="README.md")
        self.read_tool.run(file_path="package.json")
        self.guardrail = ScoutGuardrail(self.ctx, max_retries=2)

    def valid(self) -> dict:
        return {
            "repository_path": str(self.repo),
            "primary_language": "JavaScript",
            "framework": "Express",
            "stack_components": ["express ^4.19.0"],
            "business_intent": "Tracks rooms and payments for hostels.",
            "critical_user_flows": ["record payment"],
            "sensitive_data": [],
            "entry_points": ["src/server.js"],
            "key_files": ["src/server.js", "src/app.py"],
            "evidence": [
                {"file": "README.md", "lines": "3", "quote": "Tracks rooms and payments for hostels.", "fact": "intent"},
                {"file": "package.json", "lines": "3", "quote": '"dependencies": {"express": "^4.19.0"}', "fact": "uses express"},
            ],
        }

    def test_accepts_verified_output(self):
        ok, result = self.guardrail(output("```json\n" + json.dumps(self.valid()) + "\n```"))
        self.assertTrue(ok)
        self.assertEqual(len(json.loads(result)["evidence"]), 2)
        self.assertFalse(self.ctx.is_degraded)

    def test_rejects_file_not_in_ledger(self):
        data = self.valid()
        data["evidence"].append(
            {"file": "src/app.py", "lines": "4", "quote": "class Ledger:", "fact": "never read"}
        )
        ok, message = self.guardrail(output(data))
        self.assertFalse(ok)
        self.assertIn("src/app.py", message)
        self.assertIn("never read", message)

    def test_rejects_fabricated_quote(self):
        data = self.valid()
        data["evidence"][0]["quote"] = "Tracks rooms and payments for hospitals."
        ok, message = self.guardrail(output(data))
        self.assertFalse(ok)
        self.assertIn("does not match", message)

    def test_rejects_key_file_that_does_not_exist(self):
        data = self.valid()
        data["key_files"].append("src/payments/stripe.js")
        ok, message = self.guardrail(output(data))
        self.assertFalse(ok)
        self.assertIn("src/payments/stripe.js", message)

    def test_rejects_non_json_and_old_string_evidence(self):
        self.assertFalse(self.guardrail(output("I reviewed everything, looks fine."))[0])
        data = self.valid()
        data["evidence"] = ["README.md: it is a hostel app"]
        self.assertFalse(self.guardrail(output(data))[0])

    def test_retry_then_accept_does_not_degrade(self):
        bad = self.valid()
        bad["evidence"][0]["quote"] = "This sentence is not in the README at all."
        self.assertFalse(self.guardrail(output(bad))[0])
        ok, _ = self.guardrail(output(self.valid()))
        self.assertTrue(ok)
        self.assertFalse(self.ctx.is_degraded)
        self.assertTrue(self.ctx.writes_enabled)

    def test_degrades_after_retries_and_strips_unverified_claims(self):
        bad = self.valid()
        bad["evidence"][1] = {
            "file": "src/app.py", "lines": "1", "quote": "import os  # never read", "fact": "x",
        }
        bad["key_files"].append("ghost.py")
        for _ in range(2):
            self.assertFalse(self.guardrail(output(bad))[0])
        ok, result = self.guardrail(output(bad))
        self.assertTrue(ok, "exhausted retries must degrade, not fail")
        cleaned = json.loads(result)
        self.assertEqual([e["file"] for e in cleaned["evidence"]], ["README.md"])
        self.assertNotIn("ghost.py", cleaned["key_files"])
        self.assertIn(DEGRADED_KEY, cleaned)
        self.assertTrue(self.ctx.is_degraded)
        self.assertFalse(self.ctx.writes_enabled)
        self.assertEqual(len(self.ctx.degraded_reasons), 1)

    def test_degrades_to_fallback_when_output_never_parses(self):
        for _ in range(2):
            self.assertFalse(self.guardrail(output("not json"))[0])
        ok, result = self.guardrail(output("still not json"))
        self.assertTrue(ok)
        cleaned = json.loads(result)
        self.assertEqual(cleaned["framework"], "unknown")
        self.assertEqual(cleaned["evidence"], [])
        self.assertTrue(self.ctx.is_degraded)


class TestOptimizerGuardrail(RepoFixture):
    def setUp(self):
        super().setUp()
        self.read_tool.run(file_path="src/app.py")
        self.guardrail = OptimizerGuardrail(self.ctx, max_retries=2)

    def valid(self) -> dict:
        return {
            "files_reviewed": ["src/app.py"],
            "findings": [
                {
                    "file": "src/app.py",
                    "lines": "10-13",
                    "symbol": "Ledger.find_duplicates",
                    "issue": "nested loop",
                    "offending_code": "for a in items:\n    for b in items:\n        if a is not b and a == b:",
                    "current_complexity": "O(N^2)",
                    "target_complexity": "O(N)",
                    "rewrite": "use a Counter",
                    "impact": "high",
                }
            ],
            "bloat": [{"file": "src/app.py", "lines": "1", "what_to_delete_or_merge": "unused import os"}],
            "summary": "One quadratic loop.",
        }

    def test_accepts_verified_output(self):
        ok, result = self.guardrail(output(self.valid()))
        self.assertTrue(ok)
        self.assertEqual(len(json.loads(result)["findings"]), 1)

    def test_rejects_unread_reviewed_file(self):
        data = self.valid()
        data["files_reviewed"].append("src/server.js")
        ok, message = self.guardrail(output(data))
        self.assertFalse(ok)
        self.assertIn("src/server.js", message)

    def test_rejects_finding_in_unread_file(self):
        data = self.valid()
        data["findings"][0]["file"] = "src/server.js"
        self.assertFalse(self.guardrail(output(data))[0])

    def test_rejects_fabricated_offending_code(self):
        data = self.valid()
        data["findings"][0]["offending_code"] = "for i in range(len(items)):\n    items.index(items[i])"
        ok, message = self.guardrail(output(data))
        self.assertFalse(ok)
        self.assertIn("offending_code", message)

    def test_rejects_lines_that_were_never_served(self):
        data = self.valid()
        data["findings"][0]["lines"] = "400-420"
        ok, message = self.guardrail(output(data))
        self.assertFalse(ok)
        self.assertIn("400-420", message)

    def test_degrades_and_keeps_only_verified_findings(self):
        data = self.valid()
        fake = dict(data["findings"][0], offending_code="while True:\n    hallucinated_call()")
        data["findings"].append(fake)
        data["files_reviewed"].append("src/server.js")
        for _ in range(2):
            self.assertFalse(self.guardrail(output(data))[0])
        ok, result = self.guardrail(output(data))
        self.assertTrue(ok)
        cleaned = json.loads(result)
        self.assertEqual(len(cleaned["findings"]), 1)
        self.assertEqual(cleaned["files_reviewed"], ["src/app.py"])
        self.assertEqual(len(cleaned[DEGRADED_KEY]["removed_claims"]), 2)
        self.assertTrue(self.ctx.is_degraded)


class TestDegradedWriteGate(RepoFixture):
    ARGS = dict(
        vulnerability_type="race_condition",
        stack_trigger="Express double submit",
        binary_flags_dict={"data_loss_risk": True},
        fix_recommendation="Use an idempotency key.",
        confidence=0.8,
        evidence=[{"file": "src/app.py", "lines": "13", "quote": "duplicates.append(a)"}],
    )

    def test_insert_is_skipped_when_run_is_degraded(self):
        self.ctx.mark_degraded("test")
        set_active_run(self.ctx)
        with mock.patch.object(db_tools, "_connection", side_effect=AssertionError("database was touched")):
            result = json.loads(db_tools.InsertDiscoveryTool.run(**self.ARGS))
        self.assertEqual(result["status"], "skipped")

    def test_insert_proceeds_and_carries_run_id_when_healthy(self):
        self.ctx.db_run_id = 77
        self.read_tool.run(file_path="src/app.py")
        set_active_run(self.ctx)
        cursor = mock.MagicMock()
        cursor.fetchone.return_value = (5,)
        connection = mock.MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        manager = mock.MagicMock()
        manager.__enter__.return_value = connection
        with mock.patch.object(db_tools, "_connection", return_value=manager):
            result = json.loads(db_tools.InsertDiscoveryTool.run(**self.ARGS))
        self.assertEqual(result["status"], "inserted")
        sql, params = cursor.execute.call_args.args
        self.assertIn("run_id", sql)
        self.assertEqual(params[-1], 77)

    def test_run_tracking_failure_does_not_raise(self):
        with mock.patch.object(db_tools, "_connection", side_effect=RuntimeError("db down")):
            self.assertIsNone(db_tools.start_analysis_run("abc", str(self.repo)))
            self.assertFalse(db_tools.finish_analysis_run(1, "failed", {}, []))


class TestCrewIntegration(RepoFixture):
    def test_guardrails_are_accepted_by_crewai_tasks(self):
        from crewai import Task

        for guardrail in (ScoutGuardrail(self.ctx), OptimizerGuardrail(self.ctx)):
            task = Task(
                description="d",
                expected_output="e",
                guardrail=guardrail.as_task_guardrail(),
                guardrail_max_retries=guardrail.crew_max_retries,
            )
            self.assertEqual(task.guardrail_max_retries, 3)

    def test_schema_migration_is_additive(self):
        schema = (Path(db_tools.SCHEMA_PATH)).read_text(encoding="utf-8").upper()
        self.assertNotIn("DROP TABLE", schema)
        self.assertNotIn("DROP COLUMN", schema)
        self.assertNotIn("TRUNCATE", schema)
        self.assertIn("CREATE TABLE IF NOT EXISTS ANALYSIS_RUNS", schema)
        self.assertIn("ADD COLUMN IF NOT EXISTS RUN_ID", schema)
        self.assertIn("FOREIGN KEY (RUN_ID) REFERENCES ANALYSIS_RUNS", schema)


class TestParsers(unittest.TestCase):
    def test_extract_json_object(self):
        self.assertEqual(extract_json_object('Here you go:\n{"a": 1}\nThanks'), {"a": 1})
        self.assertEqual(extract_json_object('```json\n{"a": {"b": 2}}\n```'), {"a": {"b": 2}})
        self.assertIsNone(extract_json_object("no json here"))
        self.assertIsNone(extract_json_object("[1, 2, 3]"))

    def test_parse_line_range(self):
        self.assertEqual(parse_line_range("42-57"), (42, 57))
        self.assertEqual(parse_line_range("L42-L57"), (42, 57))
        self.assertEqual(parse_line_range("lines 42 to 57"), (42, 57))
        self.assertEqual(parse_line_range("42"), (42, 42))
        self.assertIsNone(parse_line_range("module level"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
