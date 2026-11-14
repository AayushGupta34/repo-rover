"""Phase 2 offline tests: the failure catalogue seed file, its validator and
the novelty check used by `catalog promote`. No network or database needed.

    uv run python -m unittest discover -s tests -v
"""

import copy
import io
import unittest
from contextlib import redirect_stdout

from repo_rover import catalog
from repo_rover.catalog import (
    BINARY_FLAG_KEYS,
    DIFFLIB_THRESHOLD,
    REQUIRED_SLUGS,
    best_match_difflib,
    load_catalog,
    normalized_flags,
    promoted_slug,
    promoted_title,
    seed_rows,
    validate_catalog,
)


def minimal_catalog() -> dict:
    """A tiny structurally valid catalogue (ignoring the size/mandatory rules)."""
    return {
        "version": 1,
        "categories": {"performance": "Slow things."},
        "entries": [
            {
                "slug": "example-entry",
                "title": "Example entry",
                "category": "performance",
                "vulnerability_type": "example_issue",
                "stack_trigger": "Express",
                "binary_flags": {"user_visible": True},
                "symptom": "Something observable goes wrong for the user under load.",
                "why_ai_makes_this_mistake": "The shortest working code omits the safeguard entirely.",
                "detection": [
                    {"type": "regex", "pattern": r"foo\(", "should_match": ["foo(1)"], "should_not_match": ["bar(1)"]}
                ],
                "fix_recommendation": "Add the safeguard and verify it with a regression test.",
                "verification_steps": ["Run the regression test."],
                "references": [{"title": "Some standard", "url": None}],
            }
        ],
    }


def structural_errors(data: dict) -> list[str]:
    """Validator errors excluding the whole-catalogue size and mandatory rules."""
    errors, _ = validate_catalog(data)
    return [e for e in errors if not e.startswith("catalogue:")]


class TestSeedFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_catalog(catalog.DEFAULT_CATALOG_PATH)
        cls.errors, cls.report = validate_catalog(cls.data)

    def test_seed_file_is_valid(self):
        self.assertEqual(self.errors, [])

    def test_size_and_detector_targets(self):
        self.assertGreaterEqual(self.report["entries"], 100)
        self.assertLessEqual(self.report["entries"], 160)
        self.assertGreaterEqual(self.report["deterministic_entries"], 60)
        self.assertGreater(self.report["regex_samples_checked"], 100)

    def test_mandatory_case_study_entries_present(self):
        slugs = {entry["slug"] for entry in self.data["entries"]}
        for slug in REQUIRED_SLUGS:
            self.assertIn(slug, slugs)

    def test_every_category_is_populated(self):
        self.assertEqual(set(self.report["categories"]), set(self.data["categories"]))

    def test_reference_urls_are_restricted_to_known_stable_sources(self):
        allowed = ("https://cwe.mitre.org/data/definitions/", "https://owasp.org/Top10/")
        for entry in self.data["entries"]:
            for reference in entry.get("references") or []:
                url = reference.get("url")
                if url is not None:
                    self.assertTrue(url.startswith(allowed), f"{entry['slug']}: {url}")

    def test_flag_vocabulary_matches_the_analysis_crew(self):
        from repo_rover import analysis_crew

        self.assertEqual(tuple(analysis_crew.BINARY_FLAG_KEYS), tuple(BINARY_FLAG_KEYS))

    def test_seed_rows_match_the_upsert_columns(self):
        rows = seed_rows(self.data)
        self.assertEqual(len(rows), self.report["entries"])
        self.assertTrue(all(len(row) == len(catalog.SEED_COLUMNS) for row in rows))
        flags = rows[0][catalog.SEED_COLUMNS.index("binary_flags")].adapted
        self.assertEqual(set(flags), set(BINARY_FLAG_KEYS))


class TestValidator(unittest.TestCase):
    def test_minimal_catalogue_is_structurally_valid(self):
        self.assertEqual(structural_errors(minimal_catalog()), [])

    def assert_rejected(self, mutate, expected_fragment: str):
        data = minimal_catalog()
        mutate(data)
        errors = structural_errors(data)
        self.assertTrue(
            any(expected_fragment in error for error in errors),
            f"expected an error containing {expected_fragment!r}, got {errors}",
        )

    def test_duplicate_slug(self):
        self.assert_rejected(lambda d: d["entries"].append(copy.deepcopy(d["entries"][0])), "duplicate slug")

    def test_bad_slug_format(self):
        self.assert_rejected(lambda d: d["entries"][0].update(slug="Not A Slug"), "kebab-case")

    def test_missing_required_field(self):
        self.assert_rejected(lambda d: d["entries"][0].pop("symptom"), "required field 'symptom'")

    def test_empty_required_field(self):
        self.assert_rejected(lambda d: d["entries"][0].update(fix_recommendation="  "), "fix_recommendation")

    def test_padded_short_prose(self):
        self.assert_rejected(lambda d: d["entries"][0].update(symptom="Bad."), "too short")

    def test_unknown_field(self):
        self.assert_rejected(lambda d: d["entries"][0].update(severity="high"), "unknown field")

    def test_undeclared_category(self):
        self.assert_rejected(lambda d: d["entries"][0].update(category="misc"), "not declared")

    def test_unknown_flag_and_non_boolean_flag(self):
        self.assert_rejected(lambda d: d["entries"][0].update(binary_flags={"is_bad": True}), "unknown binary flag")
        self.assert_rejected(lambda d: d["entries"][0].update(binary_flags={"user_visible": "yes"}), "must be true or false")

    def test_regex_that_does_not_compile(self):
        self.assert_rejected(
            lambda d: d["entries"][0].update(detection=[{"type": "regex", "pattern": "("}]), "does not compile"
        )

    def test_regex_self_tests_are_enforced(self):
        self.assert_rejected(
            lambda d: d["entries"][0]["detection"][0].update(should_match=["bar(1)"]), "does not match"
        )
        self.assert_rejected(
            lambda d: d["entries"][0]["detection"][0].update(should_not_match=["foo(2)"]), "wrongly matches"
        )

    def test_detector_type_and_shape(self):
        self.assert_rejected(lambda d: d["entries"][0].update(detection=[{"type": "magic"}]), "detector type")
        self.assert_rejected(lambda d: d["entries"][0].update(detection=[{"type": "llm_only"}]), "question")
        self.assert_rejected(
            lambda d: d["entries"][0].update(detection=[{"type": "manifest", "manifest_files": ["package.json"]}]),
            "has_any",
        )
        self.assert_rejected(lambda d: d["entries"][0].update(detection=[]), "detection")

    def test_reference_url_must_be_null_or_https(self):
        self.assert_rejected(
            lambda d: d["entries"][0].update(references=[{"title": "x", "url": "http://insecure.example"}]),
            "https://",
        )

    def test_whole_catalogue_rules(self):
        errors, _ = validate_catalog(minimal_catalog())
        self.assertTrue(any("mandatory entry 'no-rate-limiting'" in e for e in errors))
        self.assertTrue(any("deterministic detector" in e for e in errors))


class TestNoveltyAndPromotion(unittest.TestCase):
    CATALOG = [
        {
            "id": 1,
            "slug": "no-rate-limiting",
            "vulnerability_type": "missing_rate_limiting",
            "stack_trigger": "Express, Fastify, Next.js API routes, FastAPI, Flask, Django, any public API",
            "fix_recommendation": "Apply per-IP and per-user limits backed by a shared store and return 429.",
        },
        {
            "id": 2,
            "slug": "floating-point-money",
            "vulnerability_type": "float_currency_arithmetic",
            "stack_trigger": "JavaScript Number, Python float, SQL FLOAT and REAL columns",
            "fix_recommendation": "Store money as integer minor units or a decimal type.",
        },
    ]

    def test_near_duplicate_scores_above_threshold(self):
        discovery = {
            "id": 9,
            "vulnerability_type": "missing_rate_limiting",
            "stack_trigger": "Express, Fastify, Next.js API routes, Flask, any public API",
            "fix_recommendation": "Apply per-IP and per-user limits with a shared store; return 429.",
        }
        match, score = best_match_difflib(discovery, self.CATALOG)
        self.assertEqual(match["slug"], "no-rate-limiting")
        self.assertGreaterEqual(score, DIFFLIB_THRESHOLD)

    def test_novel_discovery_scores_below_threshold(self):
        discovery = {
            "id": 10,
            "vulnerability_type": "bluetooth_pairing_replay",
            "stack_trigger": "React Native BLE peripheral reconnect after app backgrounding",
            "fix_recommendation": "Bind the session key to a fresh nonce on every reconnect.",
        }
        _, score = best_match_difflib(discovery, self.CATALOG)
        self.assertLess(score, DIFFLIB_THRESHOLD)

    def test_empty_catalogue_means_everything_is_novel(self):
        match, score = best_match_difflib({"id": 1, "vulnerability_type": "x", "stack_trigger": "y"}, [])
        self.assertIsNone(match)
        self.assertEqual(score, 0.0)

    def test_promoted_slug_and_title_are_valid(self):
        discovery = {"id": 42, "vulnerability_type": "Race_Condition!! (double submit)", "stack_trigger": "Express"}
        slug = promoted_slug(discovery)
        self.assertEqual(slug, "disc-42-race-condition-double-submit")
        self.assertRegex(slug, catalog.SLUG_PATTERN)
        self.assertIn("Express", promoted_title(discovery))

    def test_normalized_flags_fill_every_key(self):
        flags = normalized_flags({"user_visible": True})
        self.assertEqual(set(flags), set(BINARY_FLAG_KEYS))
        self.assertTrue(flags["user_visible"])
        self.assertFalse(flags["data_loss_risk"])


class TestCli(unittest.TestCase):
    def run_cli(self, *argv: str) -> tuple[int, str]:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = catalog.main(["catalog", *argv])
        return code, buffer.getvalue()

    def test_validate_command_passes_on_the_seed_file(self):
        code, out = self.run_cli("validate")
        self.assertEqual(code, 0)
        self.assertIn("VALID", out)

    def test_stats_command_prints_counts_per_category(self):
        code, out = self.run_cli("stats")
        self.assertEqual(code, 0)
        for category in ("performance", "auth_access", "ai_llm_integration", "TOTAL"):
            self.assertIn(category, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
