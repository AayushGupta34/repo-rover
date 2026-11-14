"""Chaos framework: experiments, fault injectors ("monkeys") and the runner.

An Experiment states a hypothesis about how a system behaves under one fault.
The ChaosRunner checks the steady state, injects the fault, observes, judges
the hypothesis, and always rolls the fault back, even if anything raises.

Safety model
- Every experiment declares a blast radius: the only targets it may touch and
  a maximum duration. A target outside the blast radius, or one that is not
  local, is refused before anything is injected.
- Monkeys act on a named in-process seam (an attribute holding a callable).
  They only wrap that callable and restore it on rollback.
- The runner has a hard global timeout and a seed for reproducible randomness.
"""

from __future__ import annotations

import ipaddress
import json
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlsplit

from repo_rover.resilience import SimulatedHTTPError

IN_PROCESS_PREFIX = "in-process:"
LOCAL_HOSTNAMES = frozenset({"localhost", "host.docker.internal", "0.0.0.0"})
DOCKER_BRIDGE_NETWORK = ipaddress.ip_network("172.16.0.0/12")

STATUS_PASSED = "passed"      # hypothesis held and the system recovered
STATUS_FAILED = "failed"      # hypothesis was disproved, or no recovery
STATUS_ABORTED = "aborted"    # stopped early: timeout or an abort condition
STATUS_REFUSED = "refused"    # target outside the blast radius or not local
STATUS_SKIPPED = "skipped"    # global timeout reached before it could start
STATUS_ERROR = "error"        # the experiment itself is broken


def is_local_target(target: str, allow_docker_bridge: bool = False) -> bool:
    """True only for in-process seams and addresses on this machine.

    Accepted: `in-process:<name>`, localhost, *.localhost, loopback IPs,
    host.docker.internal, and (only with allow_docker_bridge) addresses in
    Docker's default bridge range. Everything else is refused, including names
    that merely start with "localhost" and URLs with embedded credentials.
    """
    text = str(target or "").strip()
    if not text:
        return False
    if text.startswith(IN_PROCESS_PREFIX):
        return len(text) > len(IN_PROCESS_PREFIX)
    try:
        parts = urlsplit(text if "://" in text else f"//{text}")
        host = (parts.hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    if not host or parts.username or parts.password:
        return False
    if parts.scheme and parts.scheme not in ("http", "https", "ws", "wss"):
        return False
    if host in LOCAL_HOSTNAMES or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    return allow_docker_bridge and address in DOCKER_BRIDGE_NETWORK


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class ProbeResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Probe:
    """A steady-state check: returns True (or a ProbeResult) when healthy."""

    name: str
    check: Callable[[], Any]

    def run(self) -> ProbeResult:
        try:
            value = self.check()
        except Exception as exc:
            return ProbeResult(self.name, False, f"{type(exc).__name__}: {exc}")
        if isinstance(value, ProbeResult):
            return value
        if isinstance(value, tuple):
            return ProbeResult(self.name, bool(value[0]), str(value[1]))
        return ProbeResult(self.name, bool(value))


@dataclass
class BlastRadius:
    allowed_targets: list[str]
    max_duration_seconds: float = 30.0


@dataclass
class FaultSpec:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AbortCondition:
    """Checked against the observation; True means stop and report aborted."""

    name: str
    triggered: Callable[[Any], bool]


@dataclass
class ChaosContext:
    seed: int
    rng: random.Random
    deadline: float
    events: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        self.events.append(message)


@dataclass
class Experiment:
    id: str
    name: str
    hypothesis: str
    fault: FaultSpec
    blast_radius: BlastRadius
    action: Callable[[ChaosContext], Any]
    evaluate: Callable[[Any, ChaosContext], Any]
    targets: list[str] = field(default_factory=list)
    steady_state: list[Probe] = field(default_factory=list)
    monkeys: list["Monkey"] = field(default_factory=list)
    abort_conditions: list[AbortCondition] = field(default_factory=list)
    rollback: Callable[[], None] | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "hypothesis": self.hypothesis,
            "steady_state_probes": [p.name for p in self.steady_state],
            "fault": {"type": self.fault.type, "params": self.fault.params},
            "blast_radius": {
                "allowed_targets": self.blast_radius.allowed_targets,
                "max_duration_seconds": self.blast_radius.max_duration_seconds,
            },
            "targets": self.targets,
            "abort_conditions": [c.name for c in self.abort_conditions],
            "rollback": [type(m).__name__ for m in self.monkeys] + (["custom"] if self.rollback else []),
        }


@dataclass
class ExperimentResult:
    experiment_id: str
    name: str
    hypothesis: str
    status: str = STATUS_ERROR
    detail: str = ""
    steady_before: list[ProbeResult] = field(default_factory=list)
    steady_after: list[ProbeResult] = field(default_factory=list)
    recovered: bool | None = None
    rollback_ok: bool = True
    duration_seconds: float = 0.0
    events: list[str] = field(default_factory=list)
    observation: Any = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.experiment_id,
            "name": self.name,
            "hypothesis": self.hypothesis,
            "status": self.status,
            "detail": self.detail,
            "recovered": self.recovered,
            "rollback_ok": self.rollback_ok,
            "duration_seconds": round(self.duration_seconds, 3),
            "steady_before": [vars(p) for p in self.steady_before],
            "steady_after": [vars(p) for p in self.steady_after],
            "events": self.events,
        }


# --------------------------------------------------------------------------- #
# Monkeys
# --------------------------------------------------------------------------- #
class Monkey:
    """Base fault injector: wraps one callable attribute and restores it."""

    def __init__(self, owner: Any, attribute: str):
        self.owner = owner
        self.attribute = attribute
        self.calls = 0
        self.faults_injected = 0
        self._original: Any = None
        self._active = False

    def _wrap(self, original: Callable[..., Any], ctx: ChaosContext) -> Callable[..., Any]:
        raise NotImplementedError

    def inject(self, ctx: ChaosContext) -> None:
        if self._active:
            return
        self._original = getattr(self.owner, self.attribute)
        setattr(self.owner, self.attribute, self._wrap(self._original, ctx))
        self._active = True

    def rollback(self) -> None:
        if self._active:
            setattr(self.owner, self.attribute, self._original)
            self._active = False


class NetworkMonkey(Monkey):
    """Simulates a bad network on a call: added latency, an offline link, or
    an HTTP error status (e.g. 429, 503).

    `failures` limits the fault to the first N calls; None means every call.
    """

    def __init__(
        self,
        owner: Any,
        attribute: str,
        mode: str = "latency",
        latency_seconds: float = 0.0,
        status_code: int = 503,
        failures: int | None = None,
        retry_after: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        super().__init__(owner, attribute)
        if mode not in ("latency", "offline", "http_error"):
            raise ValueError("NetworkMonkey mode must be latency, offline or http_error")
        self.mode = mode
        self.latency_seconds = latency_seconds
        self.status_code = status_code
        self.failures = failures
        self.retry_after = retry_after
        self.sleep = sleep

    def _wrap(self, original, ctx):
        def faulty(*args, **kwargs):
            self.calls += 1
            if self.failures is not None and self.faults_injected >= self.failures:
                return original(*args, **kwargs)
            self.faults_injected += 1
            if self.mode == "latency":
                self.sleep(self.latency_seconds * (0.5 + ctx.rng.random()))
                return original(*args, **kwargs)
            if self.mode == "offline":
                raise ConnectionError("chaos: network is offline")
            raise SimulatedHTTPError(self.status_code, f"chaos: HTTP {self.status_code}", self.retry_after)

        return faulty


class ResourceMonkey(Monkey):
    """Simulates an exhausted resource: after `allow_calls` successful calls,
    the next `failures` calls (None = all) raise the quota/timeout error.

    `error_factory` builds the error, e.g. a database statement-timeout.
    """

    def __init__(
        self,
        owner: Any,
        attribute: str,
        allow_calls: int = 0,
        failures: int | None = None,
        error_factory: Callable[[], BaseException] | None = None,
    ):
        super().__init__(owner, attribute)
        self.allow_calls = allow_calls
        self.failures = failures
        self.error_factory = error_factory or (
            lambda: SimulatedHTTPError(429, "chaos: quota exhausted")
        )

    def _wrap(self, original, ctx):
        def limited(*args, **kwargs):
            self.calls += 1
            exhausted = self.calls > self.allow_calls
            if exhausted and (self.failures is None or self.faults_injected < self.failures):
                self.faults_injected += 1
                raise self.error_factory()
            return original(*args, **kwargs)

        return limited


class DataMonkey(Monkey):
    """Produces oversized and malformed inputs, and can corrupt what a call
    returns. The data is generic fuzz for robustness testing: truncated or
    invalid JSON, wrong types, huge strings, awkward unicode. No payload here
    is an exploit for any system.
    """

    def __init__(
        self,
        owner: Any = None,
        attribute: str = "",
        corrupt: Callable[[Any, random.Random], Any] | None = None,
        failures: int | None = None,
        seed: int = 0,
    ):
        super().__init__(owner, attribute)
        self.corrupt = corrupt
        self.failures = failures
        self.rng = random.Random(seed)

    def _wrap(self, original, ctx):
        def corrupted(*args, **kwargs):
            self.calls += 1
            value = original(*args, **kwargs)
            if self.corrupt is None or (self.failures is not None and self.faults_injected >= self.failures):
                return value
            self.faults_injected += 1
            return self.corrupt(value, ctx.rng)

        return corrupted

    # --- generators ---------------------------------------------------------
    def oversized_text(self, size: int = 1_000_000) -> str:
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 "
        chunk = "".join(self.rng.choice(alphabet) for _ in range(1024))
        return (chunk * (size // 1024 + 1))[:size]

    def malformed_json_samples(self) -> list[str]:
        valid = json.dumps({"id": "D1", "verdict": "applies", "items": [1, 2, 3]})
        cut = self.rng.randint(1, len(valid) - 2)
        return [
            "",
            "not json at all",
            valid[:cut],                              # truncated mid-object
            "Sure! Here is the JSON you asked for:",  # prose with no object
            "```json\n" + valid[:cut] + "\n```",      # fenced and truncated
            "[1, 2, 3]",                              # valid JSON, wrong shape
            '{"judgments": "none"}',                  # wrong type for a known key
            '{"a": ' + "[" * 200 + "]" * 200 + "}",   # deep nesting
            "{" + ", ".join(f'"k{n}": null' for n in range(500)) + "}",
            '{"text": "\\ud800"}',                    # lone surrogate escape
        ]

    def fuzz_strings(self) -> list[str]:
        return [
            "",
            " " * 64,
            "\x00binary\x00",
            "../" * 12 + "outside",
            "A" * 10_000,
            "‮​﻿ mixed direction and zero width",
            "😀" * 200,
            "line1\r\nline2\rline3\n",
            "%s%s%s%n {0} {name} ${var}",
            "-1",
            "9" * 400,
            "NaN",
        ]

    @staticmethod
    def truncate(value: Any, rng: random.Random) -> str:
        text = str(getattr(value, "content", value))
        return text[: max(1, rng.randint(1, max(1, len(text) // 2)))]


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
class ChaosRunner:
    """Runs experiments safely.

    Order for each experiment: blast-radius check, steady-state probes, inject,
    act (bounded by the experiment's max duration and the global timeout),
    abort conditions, evaluate the hypothesis. Rollback runs in a `finally`
    block, and the steady state is probed again afterwards to confirm recovery.
    """

    def __init__(
        self,
        seed: int = 0,
        global_timeout_seconds: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
        allow_docker_bridge: bool = False,
    ):
        self.seed = seed
        self.global_timeout_seconds = global_timeout_seconds
        self.clock = clock
        self.allow_docker_bridge = allow_docker_bridge
        self._started = clock()

    def _global_deadline(self) -> float:
        return self._started + self.global_timeout_seconds

    def check_blast_radius(self, experiment: Experiment) -> str | None:
        """Reason the experiment must not run, or None if it is within bounds."""
        if not experiment.targets:
            return "experiment declares no targets"
        allowed = set(experiment.blast_radius.allowed_targets)
        for target in experiment.targets:
            if target not in allowed:
                return f"target '{target}' is outside the declared blast radius"
            if not is_local_target(target, self.allow_docker_bridge):
                return f"target '{target}' is not a local sandbox"
        if experiment.blast_radius.max_duration_seconds <= 0:
            return "blast radius has no positive max duration"
        return None

    @staticmethod
    def _act_with_timeout(experiment: Experiment, ctx: ChaosContext, timeout: float):
        box: dict[str, Any] = {}

        def work():
            try:
                box["value"] = experiment.action(ctx)
            except BaseException as exc:  # reported, not swallowed
                box["error"] = exc

        thread = threading.Thread(target=work, name=f"chaos-{experiment.id}", daemon=True)
        thread.start()
        thread.join(max(0.0, timeout))
        if thread.is_alive():
            return None, None, True
        return box.get("value"), box.get("error"), False

    def run(self, experiment: Experiment) -> ExperimentResult:
        result = ExperimentResult(experiment.id, experiment.name, experiment.hypothesis)
        started = self.clock()
        # Each experiment gets its own generator derived from the seed, so a
        # run is reproducible regardless of which experiments are selected.
        rng = random.Random(f"{self.seed}:{experiment.id}")
        deadline = min(self._global_deadline(), started + experiment.blast_radius.max_duration_seconds)
        ctx = ChaosContext(seed=self.seed, rng=rng, deadline=deadline)
        injected: list[Monkey] = []

        try:
            if started >= self._global_deadline():
                result.status, result.detail = STATUS_SKIPPED, "global timeout reached"
                return result
            refusal = self.check_blast_radius(experiment)
            if refusal:
                result.status, result.detail = STATUS_REFUSED, refusal
                return result

            result.steady_before = [probe.run() for probe in experiment.steady_state]
            unhealthy = [p.name for p in result.steady_before if not p.ok]
            if unhealthy:
                result.status = STATUS_ABORTED
                result.detail = "steady state not met before injection: " + ", ".join(unhealthy)
                return result

            for monkey in experiment.monkeys:
                monkey.inject(ctx)
                injected.append(monkey)

            observation, error, timed_out = self._act_with_timeout(
                experiment, ctx, deadline - self.clock()
            )
            result.observation = observation
            if timed_out:
                result.status = STATUS_ABORTED
                result.detail = "hard timeout: the action did not finish within the blast radius"
                return result
            if error is not None:
                observation = error  # an exception is an observation to judge

            for condition in experiment.abort_conditions:
                if condition.triggered(observation):
                    result.status = STATUS_ABORTED
                    result.detail = f"abort condition met: {condition.name}"
                    return result

            verdict = experiment.evaluate(observation, ctx)
            held, detail = verdict if isinstance(verdict, tuple) else (verdict, "")
            result.status = STATUS_PASSED if held else STATUS_FAILED
            result.detail = str(detail)
            return result
        except Exception as exc:
            result.status = STATUS_ERROR
            result.detail = f"{type(exc).__name__}: {exc}"
            return result
        finally:
            for monkey in reversed(injected):
                try:
                    monkey.rollback()
                except Exception as exc:
                    result.rollback_ok = False
                    result.events.append(f"rollback of {type(monkey).__name__} failed: {exc}")
            if experiment.rollback is not None and result.status not in (STATUS_REFUSED, STATUS_SKIPPED):
                try:
                    experiment.rollback()
                except Exception as exc:
                    result.rollback_ok = False
                    result.events.append(f"custom rollback failed: {exc}")
            if result.status not in (STATUS_REFUSED, STATUS_SKIPPED) and result.steady_before:
                result.steady_after = [probe.run() for probe in experiment.steady_state]
                result.recovered = all(p.ok for p in result.steady_after)
                if result.status == STATUS_PASSED and not (result.recovered and result.rollback_ok):
                    result.status = STATUS_FAILED
                    result.detail = (result.detail + " " if result.detail else "") + (
                        "System did not return to steady state after rollback."
                    )
            result.events = ctx.events + result.events
            result.duration_seconds = self.clock() - started

    def run_all(self, experiments: list[Experiment]) -> list[ExperimentResult]:
        return [self.run(experiment) for experiment in experiments]


def summarize(results: list[ExperimentResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts
