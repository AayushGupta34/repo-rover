"""`reporover` command line entry point.

    reporover analyze [target] [--resume]   run the analysis crew
    reporover init-db                       apply schema.sql to DATABASE_URL
    reporover catalog validate|stats|seed|promote
    reporover chaos self                    fault-inject RepoRover itself (offline)
    reporover chaos plan --findings FILE    generate a chaos plan for a target
                         [--run --base-url URL --i-own-this-target]
    reporover bench                         score the detector stage on bench/fixtures
    reporover demo [--record]               full pipeline offline from a recorded cassette
"""

from dotenv import load_dotenv

load_dotenv()

import argparse
import json
import sys
from pathlib import Path

from repo_rover.catalog import add_catalog_commands


def _cmd_analyze(args: argparse.Namespace) -> int:
    from repo_rover.analysis_crew import main as run_analysis

    run_analysis(target=args.target, resume=args.resume)
    return 0


def _cmd_init_db(_args: argparse.Namespace) -> int:
    from repo_rover.db_tools import init_db

    init_db()
    return 0


def _cmd_chaos_self(args: argparse.Namespace) -> int:
    from repo_rover.chaos import self_test

    results = self_test.run_self_suite(seed=args.seed, timeout_seconds=args.timeout, only=args.only)
    if args.json:
        print(json.dumps([result.to_dict() for result in results], indent=2))
    else:
        self_test.print_results(results, args.seed)
    return 0 if results and all(result.ok for result in results) else 1


def _cmd_chaos_local(args: argparse.Namespace) -> int:
    from repo_rover import demo_mode
    from repo_rover.chaos import py_harness

    try:
        outcome = py_harness.run_local_chaos(
            demo_mode.fixture_path(), seed=args.seed, timeout_seconds=args.timeout, base_url=args.base_url
        )
    except ValueError as refusal:
        print(str(refusal))
        return 2
    if args.json:
        print(json.dumps(outcome, indent=2))
    else:
        py_harness.print_scorecard(outcome)
    return 0 if outcome["executed"] == len(outcome["experiments"]) else 1


def _cmd_chaos_plan(args: argparse.Namespace) -> int:
    from repo_rover.chaos import generator, target

    try:
        data = json.loads(Path(args.findings).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read findings file {args.findings}: {exc}")
        return 1
    fingerprint = data.get("fingerprint") or {}
    directory, plan = generator.generate_plan(
        data.get("report_findings") or [],
        stack_trigger=fingerprint.get("stack_trigger") or "unknown",
        target_repo=args.repo or data.get("target_repo"),
        out_root=args.out,
    )
    print(f"Chaos plan written to {directory}")
    print(f"  evidence-backed findings: {plan['evidence_backed_findings']}  (hypotheses ignored: {plan['hypotheses_ignored']})")
    for experiment in plan["experiments"]:
        cited = ", ".join(f["id"] for f in experiment["findings"])
        print(f"  {experiment['id']}  {experiment['tool']:<10} {experiment['script']:<44} justified by {cited}")
    if not plan["experiments"]:
        print("  no experiments: no evidence-backed finding maps to a chaos template")

    try:
        results, feedback = target.run_plan(
            directory,
            args.base_url,
            run=args.run,
            i_own_this_target=args.i_own_this_target,
            allow_docker_bridge=args.docker_sandbox,
            seed=args.seed,
            timeout_seconds=args.timeout,
        )
    except target.GateRefused as refusal:
        print(str(refusal))
        return 2 if args.run else 0

    for result in results:
        print(f"[{result.status.upper():^7}] {result.experiment_id}  {result.name}: {result.detail}")
        if result.recovered is False:
            print("          target did NOT return to steady state after the experiment")
    for item in feedback:
        print(f"Feedback loop: experiment {item['experiment']} -> {item['action']}")
    return 0 if all(result.ok for result in results) else 1


def _cmd_bench(args: argparse.Namespace) -> int:
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "bench" / "benchmark.py"
    if not path.is_file():
        print(f"Benchmark not found at {path}. Run from a source checkout.")
        return 1
    spec = importlib.util.spec_from_file_location("reporover_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    forwarded = (["--json"] if args.json else []) + (["--no-write"] if args.no_write else [])
    return module.main(forwarded)


def _cmd_demo(args: argparse.Namespace) -> int:
    from repo_rover import demo_mode

    try:
        outcome = demo_mode.run_demo(record=args.record, pace=args.pace, verbose=args.verbose)
    except demo_mode.CassetteError as error:
        print(f"Demo could not run: {error}")
        return 1
    metrics = outcome["metrics"]
    return 0 if metrics["unverified_backed_findings"] == 0 and metrics["matches_recording"] else 1


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(
        prog="reporover", description="RepoRover: evidence-first repository analyzer"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze a local folder or git URL")
    analyze.add_argument("target", nargs="?", help="Local folder or git URL (prompted for if omitted)")
    analyze.add_argument("--resume", action="store_true", help="Resume the last crashed run of this folder from its checkpoint")
    analyze.set_defaults(func=_cmd_analyze)

    init_db = subparsers.add_parser("init-db", help="Apply schema.sql migrations")
    init_db.set_defaults(func=_cmd_init_db)

    add_catalog_commands(subparsers)

    chaos = subparsers.add_parser("chaos", help="Defensive chaos engineering")
    chaos_commands = chaos.add_subparsers(dest="chaos_command", required=True)

    chaos_self = chaos_commands.add_parser("self", help="Inject faults into RepoRover's own LLM and database paths (offline)")
    chaos_self.add_argument("--seed", type=int, default=0, help="Seed for reproducible randomness")
    chaos_self.add_argument("--timeout", type=float, default=120.0, help="Hard global timeout in seconds")
    chaos_self.add_argument("--only", action="append", help="Run only this experiment id (repeatable)")
    chaos_self.add_argument("--json", action="store_true", help="Print results as JSON")
    chaos_self.set_defaults(func=_cmd_chaos_self)

    chaos_local = chaos_commands.add_parser(
        "local", help="Run the stdlib chaos harness against an ephemeral local fixture server"
    )
    chaos_local.add_argument("--seed", type=int, default=0, help="Seed for reproducible randomness")
    chaos_local.add_argument("--timeout", type=float, default=120.0, help="Hard global timeout in seconds")
    chaos_local.add_argument("--base-url", help="Run against your own local server instead (localhost only)")
    chaos_local.add_argument("--json", action="store_true", help="Print results as JSON")
    chaos_local.set_defaults(func=_cmd_chaos_local)

    chaos_plan = chaos_commands.add_parser(
        "plan", help="Generate a chaos plan from a run's findings.json; execute only with --run"
    )
    chaos_plan.add_argument("--findings", required=True, help="Path to a run's findings.json")
    chaos_plan.add_argument("--repo", help="Path to the analysed repository (to discover routes)")
    chaos_plan.add_argument("--out", default="out", help="Output root (default: out)")
    chaos_plan.add_argument("--run", action="store_true", help="Execute the plan against a local sandbox")
    chaos_plan.add_argument("--base-url", help="URL of your local sandbox, e.g. http://localhost:3000")
    chaos_plan.add_argument("--i-own-this-target", action="store_true", help="Confirm you own the target")
    chaos_plan.add_argument("--docker-sandbox", action="store_true", help="Also allow Docker bridge addresses (172.16.0.0/12)")
    chaos_plan.add_argument("--seed", type=int, default=0, help="Seed passed to the scripts as CHAOS_SEED")
    chaos_plan.add_argument("--timeout", type=float, default=1800.0, help="Hard global timeout in seconds")
    chaos_plan.set_defaults(func=_cmd_chaos_plan)

    bench = subparsers.add_parser("bench", help="Score the detector stage against bench/fixtures (offline)")
    bench.add_argument("--json", action="store_true", help="Print raw results as JSON")
    bench.add_argument("--no-write", action="store_true", help="Do not write bench/RESULTS.md")
    bench.set_defaults(func=_cmd_bench)

    demo = subparsers.add_parser("demo", help="Run the full pipeline offline from recorded LLM traffic")
    demo.add_argument("--record", action="store_true", help="Record a new cassette using the configured model (needs the API key)")
    demo.add_argument("--pace", type=float, default=0.0, help="Seconds to pause before each stage (for live presenting)")
    demo.add_argument("--verbose", action="store_true", help="Also show CrewAI's own agent logs")
    demo.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
