"""Pure-stdlib chaos micro-harness, and a local fixture server to run it on.

`reporover chaos local` starts an ephemeral HTTP server on 127.0.0.1 that
wraps the `static_ledger` fixture with a small, deliberately flawed payments
API, then runs real experiments against it and reports what was measured.
No third-party tool (k6, Playwright) is needed.

Safety
- The harness refuses any base URL that is not local.
- It never follows redirects. A 3xx answer is recorded as an observation and
  the Location is not requested, so a redirect cannot steer the harness to an
  internal or external host (redirect-based SSRF).
- Proxies from the environment are ignored, so traffic stays on the machine.
- Inputs are generic robustness data: bursts, oversized bodies, malformed JSON.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from repo_rover.chaos.framework import (
    BlastRadius,
    ChaosRunner,
    DataMonkey,
    Experiment,
    ExperimentResult,
    FaultSpec,
    Probe,
    is_local_target,
)

REQUEST_TIMEOUT_SECONDS = 5.0
SPIKE_REQUESTS = 200
SPIKE_CONCURRENCY = 20
OVERSIZED_BODY_BYTES = 2 * 1024 * 1024
EXTERNAL_REDIRECT_TARGET = "http://external.invalid/steal"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: urllib then raises HTTPError for the 3xx."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LocalHttpClient:
    """Minimal HTTP client confined to one local base URL, redirects disabled."""

    def __init__(self, base_url: str, allow_docker_bridge: bool = False):
        if not is_local_target(base_url, allow_docker_bridge):
            raise ValueError(f"Refusing to target '{base_url}': the chaos harness only runs against a local sandbox.")
        self.base_url = base_url.rstrip("/")
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def request(self, method: str, path: str, body: bytes | None = None) -> dict[str, Any]:
        if not path.startswith("/") or "://" in path:
            raise ValueError("path must be a local absolute path")
        request = urllib.request.Request(self.base_url + path, data=body, method=method)
        if body is not None:
            request.add_header("Content-Type", "application/json")
        started = time.perf_counter()
        location = None
        try:
            with self._opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                status, text = response.status, response.read(4096).decode("utf-8", "replace")
        except urllib.error.HTTPError as error:
            status, text = error.code, error.read(4096).decode("utf-8", "replace")
            location = error.headers.get("Location")
        except (urllib.error.URLError, OSError, ValueError) as error:
            status, text = 0, f"{type(error).__name__}: {error}"
        return {"status": status, "ms": (time.perf_counter() - started) * 1000, "body": text, "location": location}


# --------------------------------------------------------------------------- #
# Fixture server: static_ledger plus a deliberately flawed payments API
# --------------------------------------------------------------------------- #
class FixtureServer:
    """Ephemeral 127.0.0.1 server for local chaos runs. Deliberately flawed:
    no rate limit, no body-size limit, no input validation, no idempotency,
    internal errors returned to the client, and an open redirect."""

    def __init__(self, fixture_directory: str | Path):
        index = Path(fixture_directory) / "index.html"
        page = index.read_bytes() if index.is_file() else b"<html><body>fixture</body></html>"
        payments: list[dict[str, Any]] = []
        lock = threading.Lock()
        self.payments = payments

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # keep the terminal quiet
                pass

            def _send(self, status: int, body: bytes, content_type: str = "application/json", headers=None):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = urlsplit(self.path).path
                if path in ("/", "/index.html"):
                    self._send(200, page, "text/html; charset=utf-8")
                elif path == "/api/payments":
                    with lock:
                        self._send(200, json.dumps({"count": len(payments)}).encode())
                elif path == "/redirect":
                    self._send(302, b"", headers={"Location": EXTERNAL_REDIRECT_TARGET})
                else:
                    self._send(404, b'{"error":"not found"}')

            def do_POST(self):
                if urlsplit(self.path).path != "/api/payments":
                    return self._send(404, b'{"error":"not found"}')
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)  # flaw: no size limit
                try:
                    record = json.loads(raw)   # flaw: no validation of shape or types
                    amount = float(record["amount"])
                    with lock:
                        payments.append({"student": record.get("student"), "amount": amount})  # flaw: no idempotency
                    self._send(201, json.dumps({"ok": True, "count": len(payments)}).encode())
                except Exception as exc:        # flaw: internal error detail returned with a 500
                    self._send(500, json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode())

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="chaos-fixture-server", daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self) -> "FixtureServer":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def build_experiments(client: LocalHttpClient, seed: int = 0) -> list[Experiment]:
    base = client.base_url
    radius = lambda seconds: BlastRadius([base], max_duration_seconds=seconds)
    healthy = Probe("GET / answers 200", lambda: (lambda r: (r["status"] == 200, f"HTTP {r['status']}"))(client.request("GET", "/")))
    measurements: dict[str, dict[str, Any]] = {}

    def spike(ctx):
        with ThreadPoolExecutor(max_workers=SPIKE_CONCURRENCY) as pool:
            responses = list(pool.map(lambda _i: client.request("GET", "/api/payments"), range(SPIKE_REQUESTS)))
        latencies = [r["ms"] for r in responses]
        return {
            "requests": len(responses),
            "concurrency": SPIKE_CONCURRENCY,
            "http_200": sum(1 for r in responses if r["status"] == 200),
            "http_429": sum(1 for r in responses if r["status"] == 429),
            "http_5xx": sum(1 for r in responses if r["status"] >= 500),
            "failed_connections": sum(1 for r in responses if r["status"] == 0),
            "p50_ms": round(statistics.median(latencies), 1),
            "p95_ms": round(_percentile(latencies, 0.95), 1),
        }

    def spike_verdict(obs, ctx):
        measurements["spike"] = obs
        return obs["http_429"] > 0 and obs["http_5xx"] == 0, (
            f"{obs['requests']} requests at concurrency {obs['concurrency']}: {obs['http_200']} x 200, "
            f"{obs['http_429']} x 429, {obs['http_5xx']} x 5xx, p95 {obs['p95_ms']} ms"
        )

    def malformed(ctx):
        samples = DataMonkey(seed=ctx.seed).malformed_json_samples()
        statuses = [client.request("POST", "/api/payments", sample.encode("utf-8", "replace"))["status"] for sample in samples]
        return {"cases": len(statuses), "rejected_4xx": sum(1 for s in statuses if 400 <= s < 500),
                "server_errors_5xx": sum(1 for s in statuses if s >= 500), "accepted_2xx": sum(1 for s in statuses if 200 <= s < 300)}

    def malformed_verdict(obs, ctx):
        measurements["malformed_json"] = obs
        return obs["server_errors_5xx"] == 0 and obs["accepted_2xx"] == 0, (
            f"{obs['cases']} malformed bodies: {obs['rejected_4xx']} rejected with 4xx, "
            f"{obs['server_errors_5xx']} caused a 5xx, {obs['accepted_2xx']} accepted"
        )

    def oversized(ctx):
        body = json.dumps({"student": "load", "amount": 1, "note": "x" * OVERSIZED_BODY_BYTES}).encode()
        response = client.request("POST", "/api/payments", body)
        return {"body_bytes": len(body), "status": response["status"], "ms": round(response["ms"], 1)}

    def oversized_verdict(obs, ctx):
        measurements["oversized_body"] = obs
        return obs["status"] in (400, 413), f"{obs['body_bytes']} byte body answered with HTTP {obs['status']} in {obs['ms']} ms"

    def duplicate(ctx):
        before = json.loads(client.request("GET", "/api/payments")["body"])["count"]
        body = json.dumps({"student": "Asha", "amount": 500, "request_id": f"chaos-{ctx.seed}"}).encode()
        with ThreadPoolExecutor(max_workers=2) as pool:
            statuses = [r["status"] for r in pool.map(lambda _i: client.request("POST", "/api/payments", body), range(2))]
        after = json.loads(client.request("GET", "/api/payments")["body"])["count"]
        return {"statuses": statuses, "records_created": after - before}

    def duplicate_verdict(obs, ctx):
        measurements["duplicate_submit"] = obs
        return obs["records_created"] == 1, f"the same payment sent twice created {obs['records_created']} record(s)"

    def redirect(ctx):
        response = client.request("GET", "/redirect")
        return {"status": response["status"], "location": response["location"], "followed": response["status"] not in (301, 302, 303, 307, 308)}

    def redirect_verdict(obs, ctx):
        measurements["redirect"] = obs
        return not obs["followed"], f"server answered HTTP {obs['status']} -> {obs['location']}; the harness did not follow it"

    def experiment(identifier, name, hypothesis, fault, action, verdict, seconds=30):
        return Experiment(id=identifier, name=name, hypothesis=hypothesis, fault=fault, blast_radius=radius(seconds),
                          targets=[base], steady_state=[healthy], action=action, evaluate=verdict)

    experiments = [
        experiment("L01", "Request spike from one client",
                   "A burst from a single client is throttled with HTTP 429 and causes no 5xx.",
                   FaultSpec("load_spike", {"requests": SPIKE_REQUESTS, "concurrency": SPIKE_CONCURRENCY}), spike, spike_verdict, 60),
        experiment("L02", "Malformed JSON request bodies",
                   "Malformed bodies are rejected with 4xx and never cause a 5xx.",
                   FaultSpec("malformed_input", {"kind": "json"}), malformed, malformed_verdict),
        experiment("L03", "Oversized request body",
                   "A multi-megabyte body is refused with HTTP 413 or 400.",
                   FaultSpec("oversized_input", {"bytes": OVERSIZED_BODY_BYTES}), oversized, oversized_verdict),
        experiment("L04", "Duplicate concurrent submission",
                   "The same payment submitted twice at once is recorded once.",
                   FaultSpec("duplicate_request", {"copies": 2}), duplicate, duplicate_verdict),
        experiment("L05", "Redirect to another host",
                   "A redirect answer is not followed, so the harness cannot be steered off the local target.",
                   FaultSpec("redirect", {"location": EXTERNAL_REDIRECT_TARGET}), redirect, redirect_verdict),
    ]
    for item in experiments:
        item.measurements = measurements  # shared, filled as experiments run
    return experiments


def run_local_chaos(
    fixture_directory: str | Path, seed: int = 0, timeout_seconds: float = 120.0, base_url: str | None = None
) -> dict[str, Any]:
    """Run the harness against a fresh fixture server (or a given local URL)."""

    def execute(url: str) -> dict[str, Any]:
        client = LocalHttpClient(url)
        experiments = build_experiments(client, seed)
        runner = ChaosRunner(seed=seed, global_timeout_seconds=timeout_seconds)
        results: list[ExperimentResult] = runner.run_all(experiments)
        measurements = experiments[0].measurements if experiments else {}
        return {
            "target": "ephemeral fixture server on 127.0.0.1" if base_url is None else url,
            "seed": seed,
            "experiments": [
                {**result.to_dict(), "fault": experiment.fault.type,
                 "outcome": "resilient" if result.status == "passed" else
                            "weakness confirmed" if result.status == "failed" else result.status}
                for result, experiment in zip(results, experiments)
            ],
            "measurements": measurements,
            "executed": sum(1 for r in results if r.status in ("passed", "failed")),
            "resilient": sum(1 for r in results if r.status == "passed"),
            "weaknesses_confirmed": sum(1 for r in results if r.status == "failed"),
            "recovered_after_every_experiment": all(r.recovered for r in results if r.recovered is not None),
        }

    if base_url is not None:
        return execute(base_url)
    with FixtureServer(fixture_directory) as server:
        return execute(server.base_url)


def print_scorecard(outcome: dict[str, Any]) -> None:
    print(f"Local chaos run against {outcome['target']} (seed {outcome['seed']})")
    print("-" * 78)
    for item in outcome["experiments"]:
        print(f"[{item['outcome'].upper():^18}] {item['id']}  {item['name']}")
        print(f"                     hypothesis: {item['hypothesis']}")
        print(f"                     measured:   {item['detail']}")
    print("-" * 78)
    print(f"Executed {outcome['executed']} experiments: {outcome['resilient']} resilient, "
          f"{outcome['weaknesses_confirmed']} weaknesses confirmed. "
          f"Target recovered after every experiment: {'yes' if outcome['recovered_after_every_experiment'] else 'NO'}.")
