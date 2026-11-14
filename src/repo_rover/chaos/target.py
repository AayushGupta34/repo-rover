"""Run a generated chaos plan against a local sandbox, behind an execution gate.

The gate is checked before anything touches the network:
- `--run` must be given (the default is to generate the plan only);
- `--i-own-this-target` must be given;
- the base URL must be localhost, a loopback address, `*.localhost` or
  `host.docker.internal` (Docker's bridge range only with `--docker-sandbox`).
External URLs are always refused.

Feedback loop: when an experiment leaves the target in a failure state that
the experiment did not predict (for example it never recovers), that failure
is formatted as a new failure node. If nothing similar is in the catalogue it
is queued in `dynamic_discoveries` through InsertDiscoveryTool.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from repo_rover.chaos.framework import (
    STATUS_ABORTED,
    STATUS_ERROR,
    STATUS_FAILED,
    AbortCondition,
    BlastRadius,
    ChaosContext,
    ChaosRunner,
    Experiment,
    ExperimentResult,
    FaultSpec,
    Probe,
    is_local_target,
)

OUTPUT_TAIL_CHARS = 1500


class GateRefused(Exception):
    """The execution gate did not allow the plan to run."""


def check_execution_gate(
    base_url: str | None, run: bool, i_own_this_target: bool, allow_docker_bridge: bool = False
) -> None:
    """Raise GateRefused unless every condition for executing is met."""
    if not run:
        raise GateRefused("Plan generated only. Pass --run to execute it against a local sandbox.")
    if not base_url:
        raise GateRefused("--run requires --base-url pointing at your local sandbox.")
    if not is_local_target(base_url, allow_docker_bridge):
        raise GateRefused(
            f"Refusing to run against '{base_url}': only localhost, loopback addresses, *.localhost "
            "and host.docker.internal are allowed. External URLs are never targeted."
        )
    if not i_own_this_target:
        raise GateRefused("Refusing to run without --i-own-this-target. Only test systems you own.")


def http_probe(base_url: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Steady-state probe: the target answers with a status below 500."""
    try:
        with urllib.request.urlopen(base_url, timeout=timeout) as response:
            status = response.status
    except urllib.error.HTTPError as error:
        status = error.code
    except (urllib.error.URLError, OSError, ValueError) as error:
        return False, f"no response: {error}"
    return status < 500, f"HTTP {status}"


def run_command(command: list[str], cwd: Path, env: dict[str, str], timeout: float) -> dict[str, Any]:
    """Run one script. No shell is involved; arguments are passed as a list."""
    try:
        completed = subprocess.run(
            command, cwd=str(cwd), env=env, timeout=timeout, capture_output=True, text=True, errors="replace"
        )
    except subprocess.TimeoutExpired:
        return {"returncode": None, "timed_out": True, "output": ""}
    except OSError as error:
        return {"returncode": None, "timed_out": False, "output": f"could not start: {error}"}
    output = (completed.stdout or "") + (completed.stderr or "")
    return {"returncode": completed.returncode, "timed_out": False, "output": output[-OUTPUT_TAIL_CHARS:]}


def command_for(experiment: dict[str, Any], plan_dir: Path, which: Callable[[str], str | None]) -> list[str] | None:
    script = plan_dir / experiment["script"]
    if experiment["tool"] == "k6":
        binary = which("k6")
        return [binary, "run", str(script)] if binary else None
    binary = which("npx")
    if not binary:
        return None
    return [binary, "playwright", "test", script.name, "--config", "playwright.config.js"]


def load_plan(plan_dir: str | os.PathLike) -> dict[str, Any]:
    return json.loads((Path(plan_dir) / "plan.json").read_text(encoding="utf-8"))


def unexpected_failure_node(
    experiment: dict[str, Any], result: ExperimentResult, plan: dict[str, Any]
) -> dict[str, Any] | None:
    """Describe a failure the experiment did not predict, or None.

    A script that fails its assertions while the target stays healthy is the
    predicted outcome: it confirms a finding already in the catalogue. What is
    unexpected is a target that does not come back to steady state, or a
    script that could not finish inside its blast radius.
    """
    fault = experiment["fault"]["type"]
    if not result.steady_before or not all(probe.ok for probe in result.steady_before):
        return None  # the target was already unhealthy; this experiment injected nothing
    if result.status in (STATUS_FAILED, STATUS_ABORTED) and result.recovered is False:
        kind = f"no_recovery_after_{fault}"
        observation = (
            f"After chaos experiment {experiment['id']} ({experiment['name']}) ended and its fault was "
            f"rolled back, the target did not return to steady state: "
            + "; ".join(f"{p.name}: {p.detail}" for p in result.steady_after if not p.ok)
        )
        fix = (
            "The service must recover on its own once the fault ends. Add a supervisor or restart policy, "
            "bound queues and connection pools so overload sheds load instead of wedging the process, and "
            "add a health check that the platform can act on."
        )
        flags = {"user_visible": True, "requires_network": True, "data_loss_risk": False}
    elif result.status == STATUS_ABORTED and "hard timeout" in result.detail:
        kind = f"hang_under_{fault}"
        observation = (
            f"Chaos experiment {experiment['id']} ({experiment['name']}) did not finish within "
            f"{experiment['blast_radius']['max_duration_seconds']}s: the target stopped responding in time."
        )
        fix = (
            "Requests must not hang under this fault. Add timeouts on every outbound call and on request "
            "handling, and return an error instead of waiting indefinitely."
        )
        flags = {"user_visible": True, "requires_network": True, "concurrency_related": True}
    else:
        return None
    return {
        "id": experiment["id"],
        "vulnerability_type": kind,
        "stack_trigger": plan.get("stack_trigger") or "unknown",
        "binary_flags_dict": flags,
        "fix_recommendation": fix,
        "confidence": 0.6,
        "observation": observation[:600],
    }


def queue_discovery(node: dict[str, Any], catalog_entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Queue an unexpected failure in dynamic_discoveries if it is novel."""
    from repo_rover import catalog, db_tools

    match, score = catalog.best_match_difflib(node | {"fix_recommendation": node["fix_recommendation"]}, catalog_entries)
    if match is not None and score >= catalog.DIFFLIB_THRESHOLD:
        return {"experiment": node["id"], "action": "already_in_catalogue", "closest": match.get("slug"), "similarity": round(score, 3)}
    db_tools.register_chaos_observation(node["id"], node["observation"])
    response = json.loads(
        db_tools.InsertDiscoveryTool.run(
            vulnerability_type=node["vulnerability_type"],
            stack_trigger=node["stack_trigger"],
            binary_flags_dict=node["binary_flags_dict"],
            fix_recommendation=node["fix_recommendation"],
            confidence=node["confidence"],
            evidence=[{"chaos_experiment": node["id"], "observation": node["observation"]}],
        )
    )
    return {"experiment": node["id"], "action": "queued" if "id" in response else "not_queued",
            "vulnerability_type": node["vulnerability_type"], "response": response, "similarity": round(score, 3)}


def run_plan(
    plan_dir: str | os.PathLike,
    base_url: str | None,
    run: bool = False,
    i_own_this_target: bool = False,
    allow_docker_bridge: bool = False,
    seed: int = 0,
    timeout_seconds: float = 1800.0,
    executor: Callable[..., dict[str, Any]] = run_command,
    probe: Callable[[str], tuple[bool, str]] = http_probe,
    which: Callable[[str], str | None] = shutil.which,
    catalog_entries: list[dict[str, Any]] | None = None,
    queue: Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any]] = queue_discovery,
) -> tuple[list[ExperimentResult], list[dict[str, Any]]]:
    """Execute a plan's experiments. Returns (results, feedback-loop outcomes).

    Raises GateRefused before any probe or script runs if the gate is not met.
    """
    check_execution_gate(base_url, run, i_own_this_target, allow_docker_bridge)
    plan_dir = Path(plan_dir)
    plan = load_plan(plan_dir)
    if catalog_entries is None:
        from repo_rover import catalog

        catalog_entries = catalog.load_catalog(catalog.catalog_path())["entries"]

    runner = ChaosRunner(seed=seed, global_timeout_seconds=timeout_seconds, allow_docker_bridge=allow_docker_bridge)
    env = {**os.environ, "BASE_URL": base_url, "CHAOS_SEED": str(seed)}
    results: list[ExperimentResult] = []
    feedback: list[dict[str, Any]] = []

    for spec in plan["experiments"]:
        command = command_for(spec, plan_dir, which)
        cwd = plan_dir / "playwright" if spec["tool"] == "playwright" else plan_dir
        max_duration = float(spec["blast_radius"]["max_duration_seconds"])

        def action(ctx: ChaosContext, command=command, cwd=cwd, spec=spec, max_duration=max_duration):
            if command is None:
                raise FileNotFoundError(f"{spec['tool']} is not installed; cannot run {spec['script']}")
            (cwd / "results").mkdir(exist_ok=True)
            return executor(command, cwd, env, max_duration)

        def evaluate(observation, ctx: ChaosContext, spec=spec):
            if isinstance(observation, BaseException):
                raise observation
            if observation["timed_out"]:
                return False, "script exceeded its maximum duration"
            if observation["returncode"] == 0:
                return True, "all assertions and thresholds passed"
            tail = observation["output"].strip().splitlines()[-1:] or [""]
            return False, f"script exited {observation['returncode']}: assertions or thresholds failed. {tail[0][:200]}"

        experiment = Experiment(
            id=spec["id"],
            name=spec["name"],
            hypothesis=spec["hypothesis"],
            fault=FaultSpec(spec["fault"]["type"], spec["fault"]["params"]),
            blast_radius=BlastRadius([base_url], max_duration + 30),
            targets=[base_url],
            steady_state=[Probe("target answers below HTTP 500", lambda: probe(base_url))],
            abort_conditions=[
                AbortCondition("script could not be started", lambda obs: isinstance(obs, dict) and obs["returncode"] is None and not obs["timed_out"]),
            ],
            action=action,
            evaluate=evaluate,
        )
        result = runner.run(experiment)
        results.append(result)
        if result.status == STATUS_ERROR:
            continue
        node = unexpected_failure_node(spec, result, plan)
        if node is not None:
            feedback.append(queue(node, catalog_entries))
    return results, feedback
