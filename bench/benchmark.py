"""RepoRover benchmark: `reporover bench`.

Runs the deterministic stage of the pipeline (stack fingerprint + catalogue
detectors) over the synthetic apps in `bench/fixtures/` and scores the result
against hand-labelled ground truth in each fixture's `expected.json`.

Metrics
- Precision, recall and F1 are computed per catalogue rule: a reported rule
  that is labelled present is a true positive; a reported rule that is not is
  a false positive; a labelled rule that was not reported is a false negative.
- Hallucination rate is the share of reported code locations that cannot be
  re-verified: the benchmark re-opens each cited file and checks that the cited
  line holds the cited text. This check is independent of the detector engine.

The benchmark makes no network calls and does not use the database: it reads
the catalogue from the YAML seed so results are reproducible from a checkout.
The LLM agents are not invoked. If a recorded demo cassette is present, its
replay is summarised separately as the "agent stage".
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = BENCH_DIR / "fixtures"
RESULTS_PATH = BENCH_DIR / "RESULTS.md"
# Categories that describe process and operations rather than code defects.
PROCESS_CATEGORIES = frozenset({"testing_release", "observability_ops"})


def prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def verify_hit(fixture_root: Path, hit: dict[str, Any]) -> bool | None:
    """Independently check a cited location. None if the hit cites no location."""
    if not hit.get("file") or not hit.get("line"):
        return None
    path = fixture_root / hit["file"]
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    start, end = hit["line"], hit.get("end_line") or hit["line"]
    if start < 1 or end > len(lines):
        return False
    actual = [line.strip() for line in lines[start - 1:end] if line.strip()]
    cited = [line.strip() for line in str(hit.get("snippet") or "").split("\n") if line.strip()]
    if not cited or len(cited) > len(actual):
        return False
    # A long source line may be cited truncated, so the cited text must be a prefix.
    return all(source.startswith(quote) for quote, source in zip(cited, actual))


def load_fixtures(fixtures_dir: Path = FIXTURES_DIR) -> list[tuple[str, Path, dict[str, Any]]]:
    fixtures = []
    for directory in sorted(p for p in fixtures_dir.iterdir() if p.is_dir()):
        expected_path = directory / "expected.json"
        if expected_path.is_file():
            fixtures.append((directory.name, directory, json.loads(expected_path.read_text(encoding="utf-8"))))
    return fixtures


def run_benchmark(fixtures_dir: Path = FIXTURES_DIR) -> dict[str, Any]:
    from repo_rover import catalog
    from repo_rover.detector_engine import run_detectors
    from repo_rover.fingerprint import fingerprint_repo

    entries = catalog.load_catalog(catalog.catalog_path())["entries"]
    category_of = {entry["slug"]: entry["category"] for entry in entries}
    known = set(category_of)

    per_fixture = []
    totals = {"all": [0, 0, 0], "code": [0, 0, 0]}
    located = unverified = unlocated = 0

    for name, root, expected in load_fixtures(fixtures_dir):
        labelled = {item["slug"] for item in expected["present"]}
        unknown = sorted(labelled - known)
        if unknown:
            raise ValueError(f"{name}/expected.json names rules that are not in the catalogue: {unknown}")

        fingerprint = fingerprint_repo(root)
        findings = run_detectors(root, entries, fingerprint)
        reported = {finding["slug"] for finding in findings}

        bad_locations = []
        for finding in findings:
            for hit in finding["hits"]:
                verdict = verify_hit(root, hit)
                if verdict is None:
                    unlocated += 1
                else:
                    located += 1
                    if not verdict:
                        unverified += 1
                        bad_locations.append(f"{finding['slug']} at {hit['file']}:{hit['line']}")

        def counts(slugs_filter):
            tp = len({s for s in reported & labelled if slugs_filter(s)})
            fp = len({s for s in reported - labelled if slugs_filter(s)})
            fn = len({s for s in labelled - reported if slugs_filter(s)})
            return tp, fp, fn

        scores = {
            "all": counts(lambda slug: True),
            "code": counts(lambda slug: category_of[slug] not in PROCESS_CATEGORIES),
        }
        for key, (tp, fp, fn) in scores.items():
            totals[key][0] += tp
            totals[key][1] += fp
            totals[key][2] += fn

        per_fixture.append(
            {
                "name": name,
                "description": expected.get("description", ""),
                "stack": fingerprint.stack_trigger,
                "labelled": len(labelled),
                "reported": len(reported),
                "all": prf(*scores["all"]),
                "code": prf(*scores["code"]),
                "false_positives": sorted(reported - labelled),
                "false_negatives": sorted(labelled - reported),
                "unverified_locations": bad_locations,
            }
        )

    return {
        "date": date.today().isoformat(),
        "catalogue_entries": len(entries),
        "fixtures": per_fixture,
        "overall": prf(*totals["all"]),
        "code_level": prf(*totals["code"]),
        "hallucination": {
            "located_hits": located,
            "unverified": unverified,
            "rate": unverified / located if located else 0.0,
            "hits_without_location": unlocated,
        },
        "agent_stage": agent_stage_metrics(),
    }


def agent_stage_metrics() -> dict[str, Any] | None:
    """Summary of the recorded demo cassette's replay, if one has been recorded."""
    try:
        from repo_rover import demo_mode
    except ImportError:
        return None
    if not demo_mode.cassette_path().is_file():
        return None
    try:
        outcome = demo_mode.run_demo(quiet=True, write_outputs=False)
    except Exception as exc:  # the benchmark must still report the deterministic stage
        return {"error": f"{type(exc).__name__}: {exc}"}
    return outcome["metrics"]


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def technical_strength(results: dict[str, Any]) -> str:
    overall, code, hallucination = results["overall"], results["code_level"], results["hallucination"]
    fixtures = len(results["fixtures"])
    labelled = overall["tp"] + overall["fn"]
    return (
        f"On {fixtures} synthetic fixture apps with {labelled} hand-labelled flaws, RepoRover's deterministic stage "
        f"scored {pct(overall['precision'])} precision, {pct(overall['recall'])} recall and an F1 of {overall['f1']:.3f} "
        f"({pct(code['precision'])} / {pct(code['recall'])} / {code['f1']:.3f} when process and operations rules are excluded). "
        f"Every one of the {hallucination['located_hits']} code locations it cited was re-opened and checked against the source, "
        f"giving a measured hallucination rate of {pct(hallucination['rate'])}, because findings are produced by rules over the "
        f"code and the language model is only asked to judge them. "
        f"These figures come from small fixtures written by the project's authors and measure the detector stage only, so they "
        f"show that the pipeline is grounded and reproducible, not how it performs on arbitrary real-world repositories."
    )


def render_results(results: dict[str, Any]) -> str:
    overall, code, hallucination = results["overall"], results["code_level"], results["hallucination"]

    def row(label: str, m: dict[str, Any]) -> str:
        return f"| {label} | {m['tp']} | {m['fp']} | {m['fn']} | {pct(m['precision'])} | {pct(m['recall'])} | {m['f1']:.3f} |"

    lines = [
        "# RepoRover benchmark results",
        "",
        f"Generated by `reporover bench` on {results['date']}. Reproduce with the same command; it needs no network, API key or database.",
        "",
        "## Technical Strength",
        "",
        technical_strength(results),
        "",
        "## Headline metrics",
        "",
        "| Scope | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|---|",
        row("All rules", overall),
        row("Code-level rules only", code),
        "",
        f"**Hallucination rate: {pct(hallucination['rate'])}** "
        f"({hallucination['unverified']} of {hallucination['located_hits']} cited code locations failed re-verification). "
        f"A further {hallucination['hits_without_location']} hits are absence or manifest rules that cite no single line.",
        "",
        "## Per fixture",
        "",
        "| Fixture | Stack (fingerprint) | Labelled | Reported | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for fixture in results["fixtures"]:
        m = fixture["all"]
        lines.append(
            f"| `{fixture['name']}` | {fixture['stack']} | {fixture['labelled']} | {fixture['reported']} | "
            f"{pct(m['precision'])} | {pct(m['recall'])} | {m['f1']:.3f} |"
        )
    lines += ["", "## Errors, listed in full", ""]
    for fixture in results["fixtures"]:
        lines.append(f"**`{fixture['name']}`** - {fixture['description']}")
        lines.append("- False positives (reported, judged not present): " + (", ".join(f"`{s}`" for s in fixture["false_positives"]) or "none"))
        lines.append("- False negatives (present, not reported): " + (", ".join(f"`{s}`" for s in fixture["false_negatives"]) or "none"))
        lines.append("")

    agent = results.get("agent_stage")
    lines += ["## Agent stage (recorded demo replay)", ""]
    if agent is None:
        lines.append("No cassette has been recorded, so the LLM agents were not measured. Record one with `reporover demo --record`.")
    elif "error" in agent:
        lines.append(f"The recorded cassette could not be replayed: {agent['error']}")
    else:
        lines += [
            f"Replay of the recorded run on `{agent['fixture']}` (model `{agent['model']}`, recorded {agent['recorded_at']}):",
            "",
            "| Measure | Value |",
            "|---|---|",
            f"| LLM calls replayed | {agent['llm_calls']} |",
            f"| Agent outputs rejected by guardrails and retried | {agent['guardrail_rejections']} |",
            f"| Run status | {agent['status']} |",
            f"| Findings in the report: Verified / Detector-Hit / Hypothesis | {agent['verified']} / {agent['detector_hits']} / {agent['hypotheses']} |",
            f"| Evidence-backed findings whose evidence failed re-verification | {agent['unverified_backed_findings']} |",
            f"| Discoveries written with verified evidence | {agent['discoveries_inserted']} |",
            "",
            "This is one recorded run, replayed. It shows the guardrails working on real model output; it is not a statistical sample.",
        ]
    lines += [
        "",
        "## Method and limits",
        "",
        f"- The catalogue has {results['catalogue_entries']} entries; the benchmark runs its regex, absence and manifest detectors.",
        "- Ground truth is in each fixture's `expected.json`: every flaw judged present, with its location, and every reported rule judged not present, with the reason.",
        "- Labels were written by the same people who wrote the detectors and the fixtures. Treat the scores as a regression baseline, not an independent evaluation.",
        "- Rules labelled `incidental` (for example no CI, no tests) are true of the fixtures but trivially so; the code-level row excludes process and operations rules.",
        "- False negatives are kept as found. In particular `jwt-hardcoded-secret` is missed in `express_api` because its pattern does not span an object literal; it is left unfixed here so the benchmark is not tuned to its own test set.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reporover bench", description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true", help="Print the raw results as JSON")
    parser.add_argument("--no-write", action="store_true", help="Do not write bench/RESULTS.md")
    args = parser.parse_args(argv)

    results = run_benchmark()
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        overall, hallucination = results["overall"], results["hallucination"]
        print(f"RepoRover benchmark - {len(results['fixtures'])} fixtures, {results['catalogue_entries']} catalogue entries")
        print(f"{'fixture':<16} {'labelled':>8} {'reported':>8} {'precision':>10} {'recall':>8} {'F1':>7}")
        for fixture in results["fixtures"]:
            m = fixture["all"]
            print(f"{fixture['name']:<16} {fixture['labelled']:>8} {fixture['reported']:>8} {pct(m['precision']):>10} {pct(m['recall']):>8} {m['f1']:>7.3f}")
        print(f"{'OVERALL':<16} {overall['tp'] + overall['fn']:>8} {overall['tp'] + overall['fp']:>8} {pct(overall['precision']):>10} {pct(overall['recall']):>8} {overall['f1']:>7.3f}")
        code = results["code_level"]
        print(f"{'code-level only':<16} {code['tp'] + code['fn']:>8} {code['tp'] + code['fp']:>8} {pct(code['precision']):>10} {pct(code['recall']):>8} {code['f1']:>7.3f}")
        print(f"Hallucination rate: {pct(hallucination['rate'])} ({hallucination['unverified']}/{hallucination['located_hits']} cited locations failed re-verification)")
    if not args.no_write:
        RESULTS_PATH.write_text(render_results(results), encoding="utf-8", newline="\n")
        if not args.json:
            print(f"Results written to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
