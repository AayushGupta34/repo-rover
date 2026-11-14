from dotenv import load_dotenv

load_dotenv()

import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from crewai.tools import tool

from repo_rover import resilience
from repo_rover.read_ledger import get_active_run
from repo_rover.sanitize import mask_secrets

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
MAX_QUERY_ROWS = 50
CONNECT_TIMEOUT_SECONDS = 15
STATEMENT_TIMEOUT_MS = 15_000
DB_MAX_ATTEMPTS = 4
DB_BASE_DELAY_SECONDS = 0.5
DB_MAX_DELAY_SECONDS = 8.0

# Observations recorded by the chaos runner. Chaos evidence on a discovery is
# only accepted if it matches an entry here, so an agent cannot invent one.
_chaos_observations: dict[str, str] = {}


def register_chaos_observation(experiment_id: str, observation: str) -> None:
    _chaos_observations[str(experiment_id)] = str(observation)


def with_db_retry(operation):
    """Run a whole database operation (connect, statements, commit) with
    exponential backoff and full jitter on transient failures such as lost
    connections and statement timeouts. Every operation passed here is
    idempotent, so repeating it is safe."""
    return resilience.retry_with_backoff(
        operation,
        is_retryable=resilience.is_transient_db_error,
        max_attempts=DB_MAX_ATTEMPTS,
        base_delay=DB_BASE_DELAY_SECONDS,
        max_delay=DB_MAX_DELAY_SECONDS,
    )


def repo_identifier(path: Any) -> str | None:
    """Opaque identifier for an analysed repository.

    The database is shared, so it never receives a local path, a client name
    or a clone location: only a salted-by-nothing SHA-256 prefix that lets
    rows from the same repository be grouped.
    """
    import hashlib

    if path is None:
        return None
    real = os.path.normcase(os.path.realpath(str(path)))
    return "repo:" + hashlib.sha256(real.encode("utf-8")).hexdigest()[:16]


def redact_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evidence as it is stored: secret values masked, no absolute paths."""
    cleaned = []
    for item in items:
        item = dict(item)
        for key in ("quote", "observation"):
            if key in item:
                item[key] = mask_secrets(item[key])[:600]
        if "file" in item:
            item["file"] = str(item["file"]).replace("\\", "/").lstrip("/")
        cleaned.append(item)
    return cleaned


def discovery_dedupe_key(source_repo: str | None, vulnerability_type: str, stack_trigger: str, fix: str) -> str:
    """Stable key for a discovery, so a retried or resumed insert is a no-op."""
    import hashlib

    parts = [source_repo or "", vulnerability_type, stack_trigger, re.sub(r"\s+", " ", fix)]
    return hashlib.sha256("|".join(p.strip().lower() for p in parts).encode("utf-8")).hexdigest()


@contextmanager
def _connection():
    """Open a short-lived connection to DATABASE_URL; commit on success, roll back on error.

    Neon suspends idle computes, so a fresh connection per tool call is more
    reliable than holding one open across a long crew run.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add your Neon connection string to the .env file."
        )
    conn = psycopg2.connect(database_url, connect_timeout=CONNECT_TIMEOUT_SECONDS)
    try:
        # SET LOCAL lasts for this transaction only, which also works through
        # Neon's pooled (transaction-mode) endpoint.
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (STATEMENT_TIMEOUT_MS,))
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _split_terms(stack_framework: str) -> list[str]:
    """Split 'Next.js, React / Prisma' into individual search terms."""
    terms = [t.strip() for t in re.split(r"[,;/|\n]+", stack_framework)]
    return [t for t in terms if t]


def _like_pattern(term: str) -> str:
    """Build a %term% pattern with LIKE wildcards in the term escaped."""
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _coerce_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "on"}
    return bool(value)


def _normalize_binary_flags(binary_flags: Any) -> dict[str, bool]:
    """Accept a dict or a JSON-object string and return a {flag_name: bool} dict."""
    if isinstance(binary_flags, str):
        text = binary_flags.strip()
        if not text:
            return {}
        try:
            binary_flags = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"binary_flags_dict must be a JSON object, could not parse it: {exc}"
            ) from exc
    if binary_flags is None:
        return {}
    if not isinstance(binary_flags, dict):
        raise ValueError(
            "binary_flags_dict must be a JSON object mapping flag names to true/false."
        )
    return {str(key): _coerce_flag(value) for key, value in binary_flags.items()}


@tool("QueryDatabaseTool")
def QueryDatabaseTool(stack_framework: str) -> str:
    """Look up known failure nodes for a technology stack in the Postgres database.

    Pass the framework or stack name in `stack_framework`, e.g. "Next.js" or a
    comma-separated list such as "React, Express, PostgreSQL". Returns a JSON
    object with every matching row from `known_failure_nodes` (id,
    vulnerability_type, stack_trigger, binary_flags, fix_recommendation).
    """
    terms = _split_terms(stack_framework or "")
    if not terms:
        return json.dumps(
            {"error": "stack_framework must be a non-empty framework or stack name."}
        )

    sql = """
        SELECT id, vulnerability_type, stack_trigger, binary_flags, fix_recommendation
        FROM known_failure_nodes
        WHERE stack_trigger ILIKE ANY(%(patterns)s)
           OR position(lower(stack_trigger) IN lower(%(haystack)s)) > 0
        ORDER BY id
        LIMIT %(limit)s
    """
    params = {
        "patterns": [_like_pattern(term) for term in terms],
        "haystack": " ".join(terms),
        "limit": MAX_QUERY_ROWS,
    }
    def query():
        with _connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                return [dict(row) for row in cur.fetchall()]

    try:
        rows = with_db_retry(query)
    except Exception as exc:
        return json.dumps({"error": f"Database query failed: {exc}"})

    return json.dumps(
        {"searched_terms": terms, "match_count": len(rows), "matches": rows},
        default=str,
    )


def _parse_confidence(confidence: Any) -> float:
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        raise ValueError("confidence must be a number between 0 and 1.") from None
    if not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    return value


def _parse_evidence(evidence: Any) -> list[Any]:
    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except json.JSONDecodeError as exc:
            raise ValueError(f"evidence must be a JSON array of objects: {exc}") from exc
    if isinstance(evidence, dict):
        evidence = [evidence]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(
            "evidence is required: a non-empty array of objects. Without evidence "
            "this is a hypothesis and is not written to the database."
        )
    return evidence


def _verify_source_evidence(evidence: list[Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Evidence check for callers with no analysis run (the meta crew): each
    item must name the https source it was taken from and quote it."""
    verified, problems = [], []
    for index, item in enumerate(evidence):
        if isinstance(item, dict) and item.get("chaos_experiment"):
            experiment_id = str(item["chaos_experiment"])
            observation = str(item.get("observation") or "")
            if _chaos_observations.get(experiment_id) != observation or not observation:
                problems.append(
                    f"evidence[{index}]: '{experiment_id}' is not an observation recorded by the chaos runner"
                )
            else:
                verified.append(
                    {"kind": "chaos", "experiment": experiment_id, "observation": observation[:600]}
                )
            continue
        url = item.get("source_url") if isinstance(item, dict) else None
        quote = str(item.get("quote") or "").strip() if isinstance(item, dict) else ""
        if not isinstance(url, str) or not url.startswith("https://") or not quote:
            problems.append(
                f"evidence[{index}]: needs 'source_url' (https) and a 'quote' copied from that page"
            )
            continue
        verified.append({"kind": "source", "source_url": url, "quote": quote[:600]})
    return verified, problems


@tool("InsertDiscoveryTool")
def InsertDiscoveryTool(
    vulnerability_type: str,
    stack_trigger: str,
    binary_flags_dict: dict[str, Any] | str,
    fix_recommendation: str,
    confidence: float,
    evidence: list[dict[str, Any]] | str,
) -> str:
    """Insert one evidence-backed discovery into the `dynamic_discoveries` staging table.

    Arguments:
    - vulnerability_type: short snake_case category, e.g. "race_condition".
    - stack_trigger: comma-separated component names taken ONLY from the stack fingerprint you were given, e.g. "Express, PostgreSQL".
    - binary_flags_dict: JSON object of true/false flags, e.g. {"data_loss_risk": true}.
    - fix_recommendation: concrete, actionable fix text.
    - confidence: number from 0 to 1.
    - evidence: non-empty JSON array. Each item is either {"file": "<path>", "lines": "10-12", "quote": "<text copied exactly from a tool output or detector hit>"} or {"detector": "<detector hit id such as D3>"}.
    The insert is refused if the evidence cannot be verified, if the stack is not in the fingerprint, or if the run is degraded. A refused scenario must be reported as a hypothesis with discovery_id null.
    """
    vulnerability_type = (vulnerability_type or "").strip()
    stack_trigger = (stack_trigger or "").strip()
    fix_recommendation = (fix_recommendation or "").strip()

    missing = [
        name
        for name, value in (
            ("vulnerability_type", vulnerability_type),
            ("stack_trigger", stack_trigger),
            ("fix_recommendation", fix_recommendation),
        )
        if not value
    ]
    if missing:
        return json.dumps(
            {"error": f"Missing required non-empty argument(s): {', '.join(missing)}"}
        )

    try:
        flags = _normalize_binary_flags(binary_flags_dict)
        confidence_value = _parse_confidence(confidence)
        evidence_items = _parse_evidence(evidence)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    run = get_active_run()
    skipped = json.dumps(
        {
            "status": "skipped",
            "reason": (
                f"Run is {run.status if run is not None else ''}: database writes are disabled for "
                "claims that rest on agent output. Do not retry; report discovery_id as null."
            ),
        }
    )

    if run is not None:
        # The stack written to the database comes from the deterministic
        # fingerprint, never from the model's own description of the stack.
        if run.fingerprint is not None:
            canonical, unknown = run.fingerprint.validate_stack_trigger(stack_trigger)
            if canonical is None:
                return json.dumps(
                    {
                        "error": (
                            "stack_trigger contains terms that are not in this "
                            f"repository's fingerprint: {', '.join(unknown)}."
                        ),
                        "allowed_components": sorted(set(run.fingerprint.allowed_terms().values())),
                    }
                )
            stack_trigger = canonical
        verified, problems = run.verify_evidence(evidence_items)
        label = run.evidence_label(verified)
    else:
        verified, problems = _verify_source_evidence(evidence_items)
        if any(item["kind"] == "chaos" for item in verified):
            label = "Chaos-Observed"
        else:
            label = "Source" if verified else "Hypothesis"

    if problems or not verified:
        return json.dumps(
            {
                "error": (
                    "Evidence could not be verified, so nothing was inserted. "
                    "Report this scenario as a hypothesis with discovery_id null, "
                    "or cite evidence that exists."
                ),
                "problems": problems,
            }
        )

    # A degraded run means an agent's output failed validation. Findings that
    # rest only on a deterministic detector hit are still trustworthy and are
    # recorded; anything that depends on an agent's quote is not.
    if run is not None and not run.writes_enabled:
        deterministic = all(item.get("kind") == "detector" for item in verified)
        if not (run.is_degraded and deterministic):
            return skipped

    verified = redact_evidence(verified)
    fix_recommendation = mask_secrets(fix_recommendation)
    run_id = run.db_run_id if run is not None else None
    source_repo = repo_identifier(run.root) if run is not None else None

    dedupe_key = discovery_dedupe_key(source_repo, vulnerability_type, stack_trigger, fix_recommendation)
    sql = """
        INSERT INTO dynamic_discoveries
            (vulnerability_type, stack_trigger, binary_flags, fix_recommendation,
             confidence, evidence, source_repo, dedupe_key, run_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (dedupe_key) DO UPDATE
            SET confidence = EXCLUDED.confidence,
                evidence = EXCLUDED.evidence,
                binary_flags = EXCLUDED.binary_flags,
                fix_recommendation = EXCLUDED.fix_recommendation,
                run_id = EXCLUDED.run_id
            WHERE dynamic_discoveries.review_status = 'pending'
              AND EXCLUDED.confidence > COALESCE(dynamic_discoveries.confidence, -1)
        RETURNING id, (xmax = 0) AS inserted
    """

    def insert():
        with _connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        vulnerability_type,
                        stack_trigger,
                        Json(flags),
                        fix_recommendation,
                        confidence_value,
                        Json(verified),
                        source_repo,
                        dedupe_key,
                        run_id,
                    ),
                )
                row = cur.fetchone()
                if row is not None:
                    inserted = bool(row[1]) if len(row) > 1 else True
                    return row[0], "inserted" if inserted else "updated"
                # The row exists with equal or higher confidence: a retried
                # write, a resumed run, or a weaker claim that must not replace
                # a stronger one.
                cur.execute("SELECT id FROM dynamic_discoveries WHERE dedupe_key = %s", (dedupe_key,))
                return cur.fetchone()[0], "already_recorded"

    try:
        new_id, write_status = with_db_retry(insert)
    except Exception as exc:
        return json.dumps({"error": f"Database insert failed: {exc}"})

    if run is not None:
        run.inserted_discoveries[new_id] = {
            "vulnerability_type": vulnerability_type,
            "stack_trigger": stack_trigger,
            "label": label,
            "confidence": confidence_value,
            "evidence": verified,
        }

    return json.dumps(
        {
            "status": write_status,
            "table": "dynamic_discoveries",
            "id": new_id,
            "evidence_status": label,
            "vulnerability_type": vulnerability_type,
            "stack_trigger": stack_trigger,
            "confidence": confidence_value,
            "verified_evidence_items": len(verified),
        }
    )


def start_analysis_run(run_uuid: str, target_repo: str) -> int | None:
    """Insert an analysis_runs row with status 'running' and return its id.

    Returns None (and prints a warning) if the database is unreachable or the
    migration has not been applied; run tracking must never stop an analysis.
    """
    sql = """
        INSERT INTO analysis_runs (run_uuid, target_repo, status)
        VALUES (%s, %s, 'running')
        ON CONFLICT (run_uuid) DO NOTHING
        RETURNING id
    """

    def insert():
        with _connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (run_uuid, target_repo))
                row = cur.fetchone()
                if row is None:
                    cur.execute("SELECT id FROM analysis_runs WHERE run_uuid = %s", (run_uuid,))
                    row = cur.fetchone()
                return row[0]

    try:
        return with_db_retry(insert)
    except Exception as exc:
        print(f"Warning: could not record the analysis run in the database: {exc}")
        return None


def finish_analysis_run(
    run_id: int | None,
    status: str,
    ledger_summary: dict[str, Any],
    degraded_reasons: list[str],
) -> bool:
    """Record a run's final status and read-ledger summary. Returns success."""
    if run_id is None:
        return False
    sql = """
        UPDATE analysis_runs
        SET status = %s,
            finished_at = now(),
            ledger_summary = %s,
            degraded_reasons = %s
        WHERE id = %s
    """
    def update():
        with _connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql, (status, Json(ledger_summary), Json(degraded_reasons), run_id)
                )

    try:
        with_db_retry(update)
        return True
    except Exception as exc:
        print(f"Warning: could not update the analysis run in the database: {exc}")
        return False


def init_db() -> None:
    """Create the RepoRover tables by executing schema.sql against DATABASE_URL."""
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with _connection() as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
            cur.execute(
                "SELECT (SELECT count(*) FROM known_failure_nodes), "
                "(SELECT count(*) FROM dynamic_discoveries), "
                "(SELECT count(*) FROM analysis_runs)"
            )
            known_count, discovery_count, run_count = cur.fetchone()
    print(
        f"Schema applied. known_failure_nodes: {known_count} rows, "
        f"dynamic_discoveries: {discovery_count} rows, "
        f"analysis_runs: {run_count} rows."
    )


if __name__ == "__main__":
    init_db()
