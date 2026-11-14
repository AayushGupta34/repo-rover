"""Phase 3 offline tests: the evidence-first pipeline.

Proves, without a network, an API key or a database, that:
- the stack comes from a deterministic fingerprint;
- the Inspector judges detector hits and cannot add findings of its own;
- discoveries are only written with verified evidence and a fingerprinted stack;
- the report labels every finding and keeps hypotheses in a final section.

    uv run python -m unittest discover -s tests -v
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from repo_rover import catalog, context_budget, db_tools
from repo_rover.context_budget import (
    estimate_tokens,
    max_input_tokens,
    shrink_json,
    truncate_to_tokens,
)
from repo_rover.detector_engine import format_findings_for_judge, run_detectors
from repo_rover.evidence_report import (
    HYPOTHESES_HEADING,
    InspectorGuardrail,
    RedTeamGuardrail,
    SynthesizerGuardrail,
    render_report,
)
from repo_rover.fingerprint import fingerprint_repo
from repo_rover.guardrails import DEGRADED_KEY, OptimizerGuardrail, ScoutGuardrail
from repo_rover.read_ledger import RunContext, set_active_run
from repo_rover.repo_tools import build_repo_tools

FILES = {
    "README.md": "# Hostel Ledger\n\nTracks rooms and payments for hostel managers.\n",
    "package.json": json.dumps(
        {"name": "hostel", "dependencies": {"express": "^4.19.0", "pg": "^8.11.0"}}, indent=2
    ),
    "requirements.txt": "flask==3.0.3\nrequests\n",
    "server.js": (
        'const express = require("express");\n'
        "const app = express();\n"
        "app.use(cors());\n"
        'app.get("/users/:id", async (req, res) => {\n'
        "  const rows = await db.query(`SELECT * FROM users WHERE id = ${req.params.id}`);\n"
        "  res.json(rows);\n"
        "});\n"
        "app.listen(3000);\n"
    ),
    "worker.py": (
        "import requests\n"
        "from flask import Flask\n"
        "\n"
        "def load(cursor, user_id):\n"
        '    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))\n'
        "    return requests.get(url)\n"
        "\n"
        "def pairs(items):\n"
        "    out = []\n"
        "    for a in items:\n"
        "        for b in items:\n"
        "            out.append((a, b))\n"
        "    return out\n"
    ),
    "public/index.html": (
        "<html>\n<head>\n"
        '<script src="https://cdn.tailwindcss.com"></script>\n'
        '<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.1/dist/cdn.min.js"></script>\n'
        "</head>\n<body>\n<script>\n"
        "localStorage.setItem('ledger', JSON.stringify(state));\n"
        "</script>\n</body>\n</html>\n"
    ),
    "node_modules/evil/package.json": json.dumps({"dependencies": {"django-in-node-modules": "1"}}),
    "node_modules/evil/index.js": "eval(req.query.x); const q = `SELECT * FROM t WHERE a = ${b}`;\n",
}


def output(payload) -> SimpleNamespace:
    return SimpleNamespace(raw=payload if isinstance(payload, str) else json.dumps(payload))


class PipelineFixture(unittest.TestCase):
    """A small repo with the fingerprint and detector passes already run."""

    @classmethod
    def setUpClass(cls):
        cls.entries = catalog.load_catalog(catalog.DEFAULT_CATALOG_PATH)["entries"]

    def setUp(self):
        self.workspace = Path(tempfile.mkdtemp(prefix="reporover_p3_"))
        self.addCleanup(shutil.rmtree, self.workspace, ignore_errors=True)
        self.addCleanup(set_active_run, None)
        self.repo = self.workspace / "repo"
        for relative, content in FILES.items():
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")

        self.ctx = RunContext(self.repo)
        self.ctx.fingerprint = fingerprint_repo(self.repo)
        self.findings = run_detectors(self.repo, self.entries, self.ctx.fingerprint, self.ctx)
        self.tools = build_repo_tools(self.ctx)

    def finding(self, slug: str) -> dict:
        match = [f for f in self.findings if f["slug"] == slug]
        self.assertTrue(match, f"expected detector finding {slug}; got {[f['slug'] for f in self.findings]}")
        return match[0]

    def slugs(self) -> set[str]:
        return {f["slug"] for f in self.findings}


class TestFingerprint(PipelineFixture):
    def test_components_come_from_manifests_imports_script_tags(self):
        fp = self.ctx.fingerprint
        by_source = {
            name: {e["source"] for e in component.evidence} for name, component in fp.components.items()
        }
        self.assertIn("manifest", by_source["Express"])
        self.assertIn("import", by_source["Express"])
        self.assertIn("manifest", by_source["PostgreSQL"])
        self.assertIn("manifest", by_source["Flask"])
        self.assertIn("import", by_source["requests"])
        self.assertIn("script_tag", by_source["Tailwind CSS"])
        self.assertIn("script_tag", by_source["Alpine.js"])
        self.assertIn("api_usage", by_source["localStorage"])

    def test_every_component_has_file_and_line_evidence(self):
        for component in self.ctx.fingerprint.components.values():
            self.assertTrue(component.evidence, component.name)
            for item in component.evidence:
                self.assertTrue((self.repo / item["file"]).is_file())
                self.assertGreaterEqual(item["line"], 1)

    def test_excluded_directories_do_not_influence_the_fingerprint(self):
        fp = self.ctx.fingerprint
        self.assertFalse(any("node_modules" in path for path in fp.manifests))
        self.assertNotIn("Django", fp.components)

    def test_languages_and_primary_framework(self):
        fp = self.ctx.fingerprint
        self.assertEqual(fp.languages["JavaScript"], 1)
        self.assertEqual(fp.languages["Python"], 1)
        self.assertEqual(fp.primary_framework, "Flask")
        self.assertIn("Express", fp.stack_trigger)

    def test_fingerprint_is_deterministic(self):
        self.assertEqual(fingerprint_repo(self.repo).to_dict(), self.ctx.fingerprint.to_dict())

    def test_stack_trigger_validation(self):
        fp = self.ctx.fingerprint
        self.assertEqual(fp.validate_stack_trigger("express, postgresql"), ("Express, PostgreSQL", []))
        self.assertEqual(fp.validate_stack_trigger("Alpine.js / localstorage")[0], "Alpine.js, localStorage")
        canonical, unknown = fp.validate_stack_trigger("Express, Next.js, Kubernetes")
        self.assertIsNone(canonical)
        self.assertEqual(unknown, ["Next.js", "Kubernetes"])
        self.assertIsNone(fp.validate_stack_trigger("")[0])


class TestDetectorEngine(PipelineFixture):
    def test_regex_hit_has_file_line_and_verbatim_snippet(self):
        finding = self.finding("sql-built-by-string-interpolation")
        hit = finding["hits"][0]
        self.assertEqual((hit["file"], hit["line"]), ("server.js", 5))
        source_line = FILES["server.js"].splitlines()[4].strip()
        self.assertEqual(hit["snippet"], source_line)
        self.assertEqual(finding["detector_types"], ["regex"])

    def test_safe_code_and_excluded_directories_produce_no_hits(self):
        sql_files = {h["file"] for h in self.finding("sql-built-by-string-interpolation")["hits"]}
        self.assertEqual(sql_files, {"server.js"})  # worker.py is parameterized; node_modules is skipped
        self.assertNotIn("dynamic-code-evaluation", self.slugs())  # eval() exists only in node_modules
        for finding in self.findings:
            for hit in finding["hits"]:
                self.assertNotIn("node_modules", str(hit.get("file")))

    def test_manifest_and_absence_rules_fire(self):
        rate = self.finding("no-rate-limiting")
        self.assertIn("manifest", rate["detector_types"])
        self.assertEqual(rate["hits"][0]["file"], "package.json")
        self.assertIn("express", rate["hits"][0]["snippet"])

        quota = self.finding("storage-quota-not-handled")
        self.assertEqual(quota["detector_types"], ["absence"])
        self.assertEqual(quota["hits"][0]["file"], "public/index.html")
        self.assertIn("localStorage.setItem", quota["hits"][0]["snippet"])

    def test_absence_rule_does_not_fire_when_precondition_or_safeguard_is_met(self):
        self.assertNotIn("missing-db-indexes", self.slugs())  # no schema files, so not applicable
        (self.repo / "public" / "index.html").write_text(
            FILES["public/index.html"].replace("</script>\n</body>", "/* QuotaExceededError handled */\n</script>\n</body>"),
            encoding="utf-8",
        )
        rerun = run_detectors(self.repo, self.entries, fingerprint_repo(self.repo))
        self.assertNotIn("storage-quota-not-handled", {f["slug"] for f in rerun})

    def test_other_expected_rules(self):
        for slug in ("wildcard-cors", "outbound-calls-without-timeout", "browser-storage-as-system-of-record",
                     "third-party-scripts-without-integrity", "nested-linear-lookups"):
            self.finding(slug)

    def test_findings_have_ids_and_are_registered_on_the_run(self):
        self.assertEqual([f["id"] for f in self.findings], [f"D{n}" for n in range(1, len(self.findings) + 1)])
        self.assertEqual(set(self.ctx.detector_findings), {f["id"] for f in self.findings})
        self.assertIs(self.ctx.find_detector("sql-built-by-string-interpolation"), self.finding("sql-built-by-string-interpolation"))

    def test_hits_are_quotable_evidence_without_using_the_read_budget(self):
        self.assertEqual(self.ctx.lines_served, 0)
        ok, _ = self.ctx.quote_matches("server.js", "db.query(`SELECT * FROM users WHERE id = ${req.params.id}`)")
        self.assertTrue(ok)
        self.assertFalse(self.ctx.quote_matches("server.js", "res.json(rows);")[0])  # never served

    def test_judge_block_lists_file_line_and_snippet_within_budget(self):
        block, shown = format_findings_for_judge(self.findings)
        sql = self.finding("sql-built-by-string-interpolation")
        self.assertIn(f"{sql['id']} | sql-built-by-string-interpolation", block)
        self.assertIn("server.js:5 | const rows = await db.query(", block)
        small, shown_small = format_findings_for_judge(self.findings, max_chars=600)
        self.assertLessEqual(len(small), 600 + 450)
        self.assertLess(len(shown_small), len(shown))


class TestInspectorIsAJudge(PipelineFixture):
    def setUp(self):
        super().setUp()
        self.guardrail = InspectorGuardrail(self.ctx, max_retries=2)
        self.sql = self.finding("sql-built-by-string-interpolation")
        self.cors = self.finding("wildcard-cors")

    def judgments(self, *items) -> dict:
        return {"judgments": [dict(id=i, verdict=v, severity=s, reason=r) for i, v, s, r in items]}

    def test_accepts_verdicts_on_real_hits(self):
        ok, result = self.guardrail(output(self.judgments(
            (self.sql["id"], "applies", "critical", "User id is interpolated into SQL."),
            (self.cors["id"], "not_applicable", "low", "Internal tool on a private network."),
        )))
        self.assertTrue(ok)
        summary = json.loads(result)
        self.assertEqual([f["id"] for f in summary["detector_findings"]], [self.sql["id"]])
        self.assertEqual(summary["dismissed"][0]["id"], self.cors["id"])
        registered = self.ctx.findings[self.sql["id"]]
        self.assertEqual((registered["label"], registered["severity"]), ("Detector-Hit", "critical"))
        self.assertNotIn(self.cors["id"], self.ctx.findings)

    def test_rejects_a_finding_the_engine_did_not_report(self):
        ok, message = self.guardrail(output(self.judgments(
            ("D999", "applies", "critical", "I believe there is an SSRF somewhere."),
        )))
        self.assertFalse(ok)
        self.assertIn("D999", message)
        self.assertIn("not a detector hit", message)

    def test_rejects_free_form_findings_from_the_llm(self):
        data = self.judgments((self.sql["id"], "applies", "high", "Interpolated SQL."))
        data["known_failures"] = [{"vulnerability_type": "xss", "stack_trigger": "React"}]
        ok, message = self.guardrail(output(data))
        self.assertFalse(ok)
        self.assertIn("known_failures", message)
        self.assertIn("not add findings", message)

    def test_rejects_bad_verdict_and_missing_reason(self):
        self.assertFalse(self.guardrail(output(self.judgments((self.sql["id"], "definitely", "high", "x"))))[0])
        self.assertFalse(self.guardrail(output(self.judgments((self.sql["id"], "applies", "high", ""))))[0])

    def test_invented_findings_are_stripped_after_retries_and_run_degrades(self):
        bad = self.judgments(
            (self.sql["id"], "applies", "high", "Interpolated SQL."),
            ("D999", "applies", "critical", "Invented."),
        )
        for _ in range(2):
            self.assertFalse(self.guardrail(output(bad))[0])
        ok, result = self.guardrail(output(bad))
        self.assertTrue(ok)
        self.assertTrue(self.ctx.is_degraded)
        self.assertNotIn("D999", self.ctx.findings)
        self.assertIn(DEGRADED_KEY, json.loads(result))
        self.assertTrue(all(f["source"] == "catalog" for f in self.ctx.findings.values()))

    def test_unjudged_hits_stay_as_detector_hits(self):
        ok, _ = self.guardrail(output({"judgments": []}))
        self.assertTrue(ok)
        self.assertEqual(set(self.ctx.findings), set(self.ctx.detector_findings))
        self.assertTrue(all(f["verdict"] == "unjudged" and f["label"] == "Detector-Hit" for f in self.ctx.findings.values()))

    def test_registry_only_ever_contains_engine_findings(self):
        self.guardrail(output(self.judgments((self.sql["id"], "needs_review", "medium", "Unclear."))))
        engine_slugs = {f["slug"] for f in self.findings}
        self.assertTrue({f["catalog_slug"] for f in self.ctx.findings.values()} <= engine_slugs)


class TestInsertDiscoveryEvidence(PipelineFixture):
    def setUp(self):
        super().setUp()
        self.ctx.db_run_id = 7
        set_active_run(self.ctx)
        self.cursor = mock.MagicMock()
        self.cursor.fetchone.return_value = (41,)
        connection = mock.MagicMock()
        connection.cursor.return_value.__enter__.return_value = self.cursor
        manager = mock.MagicMock()
        manager.__enter__.return_value = connection
        patcher = mock.patch.object(db_tools, "_connection", return_value=manager)
        self.connection = patcher.start()
        self.addCleanup(patcher.stop)

    def insert(self, **overrides) -> dict:
        args = dict(
            vulnerability_type="double_submit_race",
            stack_trigger="express, postgresql",
            binary_flags_dict={"data_loss_risk": True},
            fix_recommendation="Add an idempotency key with a unique constraint.",
            confidence=0.7,
            evidence=[{"file": "server.js", "lines": "5", "quote": "db.query(`SELECT * FROM users WHERE id = ${req.params.id}`)"}],
        )
        args.update(overrides)
        return json.loads(db_tools.InsertDiscoveryTool.run(**args))

    def assert_refused(self, result: dict):
        self.assertIn("error", result)
        self.connection.assert_not_called()
        self.assertEqual(self.ctx.inserted_discoveries, {})

    def test_verified_quote_is_inserted_with_confidence_evidence_and_canonical_stack(self):
        result = self.insert()
        self.assertEqual((result["status"], result["evidence_status"]), ("inserted", "Verified"))
        sql, params = self.cursor.execute.call_args.args
        self.assertIn("confidence, evidence", sql)
        self.assertEqual(params[1], "Express, PostgreSQL")
        self.assertEqual(params[4], 0.7)
        self.assertEqual(params[5].adapted[0]["kind"], "quote")
        self.assertEqual(params[-1], 7)
        self.assertEqual(self.ctx.inserted_discoveries[41]["label"], "Verified")

    def test_detector_reference_is_accepted_as_evidence(self):
        detector_id = self.finding("wildcard-cors")["id"]
        result = self.insert(evidence=[{"detector": detector_id}])
        self.assertEqual(result["evidence_status"], "Detector-Hit")

    def test_missing_evidence_is_refused(self):
        self.assert_refused(self.insert(evidence=[]))
        self.assert_refused(self.insert(evidence="[]"))

    def test_fabricated_quote_is_refused(self):
        result = self.insert(evidence=[{"file": "server.js", "quote": "await stripe.charges.create(amount)"}])
        self.assert_refused(result)
        self.assertIn("does not match", " ".join(result["problems"]))

    def test_quote_from_unread_file_and_unknown_detector_are_refused(self):
        self.assert_refused(self.insert(evidence=[{"file": "README.md", "quote": "Tracks rooms and payments"}]))
        self.assert_refused(self.insert(evidence=[{"detector": "D999"}]))

    def test_one_bad_item_blocks_the_insert(self):
        result = self.insert(evidence=[
            {"file": "server.js", "quote": "app.use(cors());"},
            {"file": "server.js", "quote": "this line is invented by the model"},
        ])
        self.assert_refused(result)

    def test_stack_outside_the_fingerprint_is_refused(self):
        result = self.insert(stack_trigger="Express, Kubernetes")
        self.assert_refused(result)
        self.assertIn("Kubernetes", result["error"])
        self.assertIn("Express", result["allowed_components"])

    def test_confidence_must_be_between_zero_and_one(self):
        self.assert_refused(self.insert(confidence=1.7))
        with self.assertRaises(ValueError):  # rejected by the tool's argument schema
            self.insert(confidence="high")
        self.connection.assert_not_called()

    def test_degraded_run_blocks_inserts(self):
        self.ctx.mark_degraded("test")
        result = self.insert()
        self.assertEqual(result["status"], "skipped")
        self.connection.assert_not_called()

    def test_without_a_run_only_sourced_evidence_is_accepted(self):
        set_active_run(None)
        refused = self.insert(evidence=[{"file": "server.js", "quote": "app.use(cors());"}])
        self.assertIn("error", refused)
        accepted = self.insert(
            stack_trigger="next 14.x",
            evidence=[{"source_url": "https://github.com/advisories", "quote": "Affected versions allow cache poisoning."}],
        )
        self.assertEqual((accepted["status"], accepted["evidence_status"]), ("inserted", "Source"))


class ReportFixture(PipelineFixture):
    """Run the guardrails in pipeline order with scripted agent outputs."""

    def setUp(self):
        super().setUp()
        self.sql = self.finding("sql-built-by-string-interpolation")
        self.cors = self.finding("wildcard-cors")
        self.tools["read"].run(file_path="README.md")
        self.tools["read"].run(file_path="worker.py")

        ScoutGuardrail(self.ctx)(output({
            "framework": "Next.js (the model's guess)", "stack_components": ["Kubernetes"],
            "business_intent": "Hostel managers track rooms and payments.",
            "critical_user_flows": ["record payment"], "entry_points": ["server.js"], "key_files": ["server.js"],
            "evidence": [
                {"file": "README.md", "lines": "3", "quote": "Tracks rooms and payments for hostel managers.", "fact": "intent"},
                {"file": "README.md", "lines": "1", "quote": "# Hostel Ledger", "fact": "name"},
            ],
        }))
        InspectorGuardrail(self.ctx)(output({"judgments": [
            {"id": self.sql["id"], "verdict": "applies", "severity": "critical", "reason": "Route parameter reaches SQL."},
            {"id": self.cors["id"], "verdict": "not_applicable", "severity": "low", "reason": "Private network only."},
        ]}))
        self.ctx.inserted_discoveries[41] = {"label": "Verified"}
        RedTeamGuardrail(self.ctx)(output({"scenarios": [
            {"title": "Ledger wiped when browser data is cleared", "narrative": "A manager clears site data and loses every payment record.",
             "vulnerability_type": "client_only_persistence", "stack_trigger": "localStorage, Alpine.js",
             "fix_recommendation": "Persist to the server.", "severity": "high", "discovery_id": 41,
             "evidence": [{"file": "public/index.html", "lines": "8", "quote": "localStorage.setItem('ledger', JSON.stringify(state));"}]},
            {"title": "CORS lets a rogue site read the ledger", "narrative": "A malicious page calls the API from a manager's browser.",
             "vulnerability_type": "cors_abuse", "stack_trigger": "Express", "fix_recommendation": "Restrict origins.",
             "severity": "medium", "discovery_id": None, "evidence": [{"detector": self.cors["id"]}]},
            {"title": "Payments double-charged during an elevator ride", "narrative": "A dropped connection causes a retried charge.",
             "vulnerability_type": "double_submit", "stack_trigger": "Stripe, Kubernetes", "fix_recommendation": "Add idempotency keys.",
             "severity": "critical", "discovery_id": 999,
             "evidence": [{"file": "server.js", "quote": "await stripe.charges.create({ amount })"}]},
        ]}))
        OptimizerGuardrail(self.ctx)(output({
            "files_reviewed": ["worker.py"],
            "findings": [{"file": "worker.py", "lines": "10-12", "symbol": "pairs", "issue": "Quadratic pairing loop",
                          "offending_code": "for a in items:\n    for b in items:\n        out.append((a, b))",
                          "current_complexity": "O(N^2)", "target_complexity": "O(N)", "rewrite": "Use itertools.", "impact": "high"}],
            "bloat": [], "summary": "One quadratic loop.",
        }))
        self.synth = SynthesizerGuardrail(self.ctx, max_retries=2)

    def render(self, items=None) -> str:
        ok, report = self.synth(output({"role": "You harden the hostel ledger.", "items": items or []}))
        self.assertTrue(ok)
        return report


class TestEvidenceLabels(ReportFixture):
    def test_stack_fields_come_from_the_fingerprint_not_the_scout(self):
        scout = self.ctx.artifacts["scout"]
        self.assertEqual(scout["framework"], "Flask")
        self.assertNotIn("Kubernetes", scout["stack_components"])
        self.assertIn("Express", scout["stack_components"])

    def test_labels_are_assigned_from_evidence(self):
        labels = {f["title"]: f["label"] for f in self.ctx.findings.values()}
        self.assertEqual(labels["SQL assembled with string interpolation"], "Detector-Hit")
        self.assertEqual(labels["Quadratic pairing loop"], "Verified")
        self.assertEqual(labels["Ledger wiped when browser data is cleared"], "Verified")
        self.assertEqual(labels["CORS lets a rogue site read the ledger"], "Detector-Hit")
        self.assertEqual(labels["Payments double-charged during an elevator ride"], "Hypothesis")

    def test_hypothesis_never_keeps_a_database_id_or_an_unfingerprinted_stack(self):
        hypothesis = next(f for f in self.ctx.findings.values() if f["label"] == "Hypothesis")
        self.assertIsNone(hypothesis["discovery_id"])
        self.assertNotIn("Kubernetes", hypothesis["stack_trigger"])
        verified = next(f for f in self.ctx.findings.values() if f["title"].startswith("Ledger wiped"))
        self.assertEqual(verified["discovery_id"], 41)


class TestSynthesizerReport(ReportFixture):
    def split(self, report: str) -> tuple[str, str]:
        self.assertEqual(report.count(HYPOTHESES_HEADING), 1)
        head, tail = report.split(HYPOTHESES_HEADING)
        return head, tail

    def numbered(self, text: str) -> list[str]:
        return [line for line in text.splitlines() if line[:1].isdigit() and ". [" in line]

    def test_every_finding_is_labelled(self):
        report = self.render()
        lines = self.numbered(report)
        self.assertEqual(len(lines), len(self.ctx.findings))
        for line in lines:
            self.assertRegex(line, r"^\d+\. \[(Verified|Detector-Hit|Hypothesis)\] ")

    def test_hypotheses_are_only_in_the_final_section(self):
        head, tail = self.split(self.render())
        self.assertNotIn("[Hypothesis] ", "\n".join(self.numbered(head)))
        self.assertNotIn("Payments double-charged", head)
        tail_items = self.numbered(tail)
        self.assertEqual(len(tail_items), 1)
        self.assertIn("[Hypothesis] Payments double-charged during an elevator ride", tail_items[0])
        self.assertIn("Evidence**: none", tail)
        self.assertNotIn("[Verified]", "\n".join(tail_items))
        self.assertNotIn("\n# ", tail)  # nothing follows the hypotheses section

    def test_work_items_carry_their_evidence(self):
        head, _ = self.split(self.render())
        self.assertIn("`server.js:5` `const rows = await db.query(", head)
        self.assertIn("quote matched exactly", head)
        self.assertIn("staged discovery id 41", head)
        self.assertIn(f"detector hit {self.cors['id']}", head)
        self.assertIn("Stack (deterministic fingerprint)", head)

    def test_dismissed_hit_is_not_reported(self):
        self.assertNotIn("CORS allows every origin", self.render())

    def test_synthesizer_wording_is_used_but_cannot_change_labels(self):
        hypothesis_id = next(f["id"] for f in self.ctx.findings.values() if f["label"] == "Hypothesis")
        report = self.render([
            {"id": self.sql["id"], "problem": "Any visitor can dump the users table.", "required_change": "Use $1 parameters.", "acceptance_check": "A quote in the id returns 400."},
            {"id": hypothesis_id, "problem": "[Verified] This definitely happens in production.", "required_change": "Add keys.", "acceptance_check": "n/a"},
        ])
        head, tail = self.split(report)
        self.assertIn("Any visitor can dump the users table.", head)
        self.assertIn("Use $1 parameters.", head)
        self.assertNotIn("This definitely happens", head)
        self.assertIn("[Hypothesis] Payments double-charged", tail)

    def test_synthesizer_cannot_add_a_finding(self):
        ok, message = self.synth(output({"role": "r", "items": [
            {"id": "X1", "problem": "Invented SSRF.", "required_change": "Fix it.", "acceptance_check": "Test."},
        ]}))
        self.assertFalse(ok)
        self.assertIn("X1", message)
        for _ in range(1):
            self.synth(output({"items": [{"id": "X1", "problem": "Invented SSRF."}]}))
        ok, report = self.synth(output({"items": [{"id": "X1", "problem": "Invented SSRF."}]}))
        self.assertTrue(ok)
        self.assertNotIn("Invented SSRF", report)
        self.assertEqual(len(self.numbered(report)), len(self.ctx.findings))

    def test_report_is_complete_even_if_the_synthesizer_writes_nothing(self):
        report = render_report(self.ctx, None)
        head, _ = self.split(report)
        self.assertIn("Use parameterized queries", head)  # catalogue's own fix text
        self.assertEqual(len(self.numbered(report)), len(self.ctx.findings))


class TestContextBudget(unittest.TestCase):
    def test_default_and_env_override(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLM_MAX_INPUT_TOKENS", None)
            self.assertEqual(max_input_tokens(), 6000)
        with mock.patch.dict(os.environ, {"LLM_MAX_INPUT_TOKENS": "2000"}):
            self.assertEqual(max_input_tokens(), 2000)
            self.assertEqual(context_budget.budget_tokens(0.5), 1000)
        with mock.patch.dict(os.environ, {"LLM_MAX_INPUT_TOKENS": "not-a-number"}):
            self.assertEqual(max_input_tokens(), 6000)

    def test_estimate_uses_four_characters_per_token(self):
        self.assertEqual(estimate_tokens("a" * 400), 100)
        self.assertEqual(estimate_tokens("a" * 401), 101)

    def test_truncate_to_tokens(self):
        text = "\n".join(f"line {n} of the tool output" for n in range(500))
        cut = truncate_to_tokens(text, 100)
        self.assertLessEqual(estimate_tokens(cut), 100)
        self.assertTrue(cut.endswith("[...truncated to fit the context budget...]"))
        self.assertEqual(truncate_to_tokens("short", 100), "short")

    def test_shrink_json_stays_valid_and_within_budget(self):
        summary = {"findings": [{"id": f"D{n}", "reason": "x" * 900} for n in range(1, 61)], "note": "y" * 5000}
        text = shrink_json(summary, 300)
        self.assertLessEqual(estimate_tokens(text), 300)
        parsed = json.loads(text)
        self.assertEqual(parsed["findings"][0]["id"], "D1")
        self.assertEqual(json.loads(shrink_json({"a": 1}, 300)), {"a": 1})

    def test_tool_output_respects_the_budget(self):
        workspace = Path(tempfile.mkdtemp(prefix="reporover_budget_"))
        self.addCleanup(shutil.rmtree, workspace, ignore_errors=True)
        (workspace / "big.py").write_text("".join(f"value_{n} = '{'x' * 60}'\n" for n in range(400)), encoding="utf-8")
        with mock.patch.dict(os.environ, {"LLM_MAX_INPUT_TOKENS": "1000"}):
            ctx = RunContext(workspace)
            result = build_repo_tools(ctx)["read"].run(file_path="big.py")
        self.assertLessEqual(estimate_tokens(result), 1000 * 0.40 + 60)
        served = ctx.entry_for("big.py").ranges()
        self.assertEqual(served[0][0], 1)
        self.assertLess(served[0][1], 200)


class TestCrewWiring(PipelineFixture):
    def test_crew_is_built_evidence_first(self):
        from repo_rover import analysis_crew

        launch = os.getcwd()
        self.addCleanup(os.chdir, launch)
        env = {"LLM_API_KEY": "test", "LLM_MODEL": "test-model", "LLM_BASE_URL": "http://127.0.0.1:1/v1",
               "DATABASE_URL": "postgresql://u:p@127.0.0.1:1/db"}
        with mock.patch.dict(os.environ, env), \
             mock.patch.object(analysis_crew, "start_analysis_run", return_value=None), \
             mock.patch.object(analysis_crew, "load_catalog_entries", return_value=(self.entries, "yaml")):
            crew = analysis_crew.run_analyzer(str(self.repo), checkpoint_directory=self.workspace)
        os.chdir(launch)
        scout, inspector, red_teamer, optimizer, synthesizer = crew.agents
        tool_names = lambda agent: [tool.name for tool in agent.tools]

        self.assertNotIn("QueryDatabaseTool", tool_names(inspector))
        self.assertEqual(tool_names(inspector), ["ReadRepoFileTool"])
        self.assertIn("InsertDiscoveryTool", tool_names(red_teamer))
        self.assertEqual(tool_names(synthesizer), [])

        scout_task, inspector_task, red_team_task, _optimizer_task, synthesis_task = crew.tasks
        self.assertIn("server.js:5 | const rows = await db.query(", inspector_task.description)
        self.assertIn("judge of these hits, not a scanner", inspector_task.description)
        self.assertIn("Express [framework] via manifest at package.json", scout_task.description)
        self.assertIn("The only terms allowed in stack_trigger are:", red_team_task.description)
        self.assertTrue(all(task.guardrail is not None for task in crew.tasks))
        self.assertLessEqual(estimate_tokens(inspector_task.description), max_input_tokens())

        run = analysis_crew.get_active_run()
        self.assertEqual(run.fingerprint.primary_framework, "Flask")
        self.assertTrue(run.detector_findings)
        report = analysis_crew.deterministic_report(run)
        self.assertIn("[Detector-Hit] SQL assembled with string interpolation", report)
        self.assertIn("not reviewed by the Inspector", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
