"""The vibe-coded failure catalogue: validate, seed, stats and promote.

`seed/failure_catalog.yaml` is the source of truth. `validate` and `stats`
work offline on that file; `seed` and `promote` talk to Postgres.

    reporover catalog validate
    reporover catalog stats [--db]
    reporover catalog seed
    reporover catalog promote [--id N ...] [--threshold T] [--dry-run]
"""

from dotenv import load_dotenv

load_dotenv()

import argparse
import difflib
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "seed" / "failure_catalog.yaml"
)

# Must stay equal to analysis_crew.BINARY_FLAG_KEYS (checked by the tests).
BINARY_FLAG_KEYS = (
    "data_loss_risk",
    "security_impact",
    "requires_network",
    "mobile_specific",
    "concurrency_related",
    "user_visible",
    "affects_payments_or_pii",
    "reproducible_offline",
)

REQUIRED_FIELDS = (
    "slug", "title", "category", "vulnerability_type", "stack_trigger",
    "binary_flags", "symptom", "why_ai_makes_this_mistake", "detection",
    "fix_recommendation", "verification_steps",
)
OPTIONAL_FIELDS = ("references", "archetypes")
ARCHETYPES = ("backend", "frontend", "static_site", "library")
AST_CHECKS = ("nested_loops", "io_in_loop", "god_function", "parser_bomb")
PROSE_FIELDS = ("symptom", "why_ai_makes_this_mistake", "fix_recommendation")
MIN_PROSE_CHARS = 40

DETECTOR_TYPES = ("regex", "absence", "manifest", "ast", "llm_only")
DETERMINISTIC_TYPES = ("regex", "absence", "manifest", "ast")
DETECTOR_KEYS = {
    "regex": {"type", "pattern", "file_globs", "should_match", "should_not_match", "note", "scan_comments"},
    "absence": {"type", "pattern", "applies_if", "applies_if_globs", "file_globs", "should_match",
                "should_not_match", "note", "scan_comments"},
    "ast": {"type", "check", "note"},
    "manifest": {"type", "manifest_files", "has_any", "lacks_all", "note"},
    "llm_only": {"type", "question"},
}

# The five entries the catalogue must always contain (startup case study).
REQUIRED_SLUGS = (
    "missing-db-indexes",
    "no-caching",
    "unoptimised-images",
    "no-load-testing",
    "no-rate-limiting",
)
MIN_DETERMINISTIC_ENTRIES = 60

SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SNAKE_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")

PROMOTED_CATEGORY = "promoted_discovery"
TRGM_THRESHOLD = 0.45
DIFFLIB_THRESHOLD = 0.60


# --------------------------------------------------------------------------- #
# Loading and validation (offline)
# --------------------------------------------------------------------------- #
def catalog_path(explicit: str | None = None) -> Path:
    return Path(explicit or os.getenv("REPOROVER_CATALOG") or DEFAULT_CATALOG_PATH)


def load_catalog(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("catalogue file must be a YAML mapping at the top level")
    return data


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_text_list(value: Any, allow_empty: bool = False) -> bool:
    if not isinstance(value, list) or (not value and not allow_empty):
        return False
    return all(_is_text(item) for item in value)


def _check_regex(where: str, name: str, pattern: Any, errors: list[str]) -> re.Pattern | None:
    if not _is_text(pattern):
        errors.append(f"{where}: '{name}' must be a non-empty regular expression")
        return None
    try:
        return re.compile(pattern, re.MULTILINE)
    except re.error as exc:
        errors.append(f"{where}: '{name}' does not compile: {exc}")
        return None


def _validate_detector(where: str, detector: Any, errors: list[str], counters: Counter) -> None:
    if not isinstance(detector, dict):
        errors.append(f"{where}: detector must be a mapping")
        return
    kind = detector.get("type")
    if kind not in DETECTOR_TYPES:
        errors.append(f"{where}: detector type must be one of {', '.join(DETECTOR_TYPES)}")
        return
    unknown = set(detector) - DETECTOR_KEYS[kind]
    if unknown:
        errors.append(f"{where}: unknown key(s) for a {kind} detector: {', '.join(sorted(unknown))}")

    if kind == "llm_only":
        if not _is_text(detector.get("question")):
            errors.append(f"{where}: llm_only detector needs a non-empty 'question'")
        return

    if kind == "ast":
        if detector.get("check") not in AST_CHECKS:
            errors.append(f"{where}: ast detector 'check' must be one of {', '.join(AST_CHECKS)}")
        return

    if kind == "manifest":
        if not _is_text_list(detector.get("manifest_files")):
            errors.append(f"{where}: manifest detector needs a non-empty 'manifest_files' list")
        has_any = detector.get("has_any", [])
        lacks_all = detector.get("lacks_all", [])
        if not _is_text_list(has_any, allow_empty=True) or not _is_text_list(lacks_all, allow_empty=True):
            errors.append(f"{where}: 'has_any' and 'lacks_all' must be lists of package names")
        elif not has_any and not lacks_all:
            errors.append(f"{where}: manifest detector needs 'has_any' and/or 'lacks_all'")
        return

    # regex / absence
    compiled = _check_regex(where, "pattern", detector.get("pattern"), errors)
    counters["regexes_compiled"] += 1
    if "applies_if" in detector:
        _check_regex(where, "applies_if", detector.get("applies_if"), errors)
        counters["regexes_compiled"] += 1
    for globs_key in ("file_globs", "applies_if_globs"):
        if globs_key in detector and not _is_text_list(detector.get(globs_key)):
            errors.append(f"{where}: '{globs_key}' must be a non-empty list of glob strings")
    if "scan_comments" in detector and not isinstance(detector["scan_comments"], bool):
        errors.append(f"{where}: 'scan_comments' must be true or false")
    for key, expected in (("should_match", True), ("should_not_match", False)):
        samples = detector.get(key, [])
        if not _is_text_list(samples, allow_empty=True):
            errors.append(f"{where}: '{key}' must be a list of sample strings")
            continue
        if compiled is None:
            continue
        for sample in samples:
            counters["regex_samples"] += 1
            if bool(compiled.search(sample)) != expected:
                verb = "does not match" if expected else "wrongly matches"
                errors.append(f"{where}: pattern {verb} {key} sample {sample!r}")


def validate_catalog(data: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Return (errors, report). An empty error list means the file is sound."""
    errors: list[str] = []
    counters: Counter = Counter()

    if not isinstance(data.get("version"), int):
        errors.append("top level: 'version' must be an integer")
    categories = data.get("categories")
    if not isinstance(categories, dict) or not categories:
        errors.append("top level: 'categories' must be a non-empty mapping of name -> description")
        categories = {}
    for name, description in categories.items():
        if not SNAKE_PATTERN.match(str(name)) or not _is_text(description):
            errors.append(f"categories: '{name}' must be snake_case with a description")
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append("top level: 'entries' must be a non-empty list")
        entries = []

    allowed_fields = set(REQUIRED_FIELDS) | set(OPTIONAL_FIELDS)
    seen_slugs: dict[str, int] = {}
    seen_titles: dict[str, str] = {}
    per_category: Counter = Counter()
    detector_types: Counter = Counter()
    deterministic_entries = 0

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"entries[{index}]: must be a mapping")
            continue
        slug = entry.get("slug")
        where = f"entries[{index}] ({slug})"

        for field in REQUIRED_FIELDS:
            value = entry.get(field)
            if value is None or value == "" or value == []:
                errors.append(f"{where}: required field '{field}' is missing or empty")
        unknown = set(entry) - allowed_fields
        if unknown:
            errors.append(f"{where}: unknown field(s): {', '.join(sorted(unknown))}")

        if _is_text(slug):
            if not SLUG_PATTERN.match(slug) or len(slug) > 80:
                errors.append(f"{where}: slug must be lowercase kebab-case, at most 80 characters")
            if slug in seen_slugs:
                errors.append(f"{where}: duplicate slug (first used by entries[{seen_slugs[slug]}])")
            seen_slugs.setdefault(slug, index)
        elif slug is not None:
            errors.append(f"{where}: slug must be a string")

        title = entry.get("title")
        if _is_text(title):
            key = title.strip().lower()
            if key in seen_titles:
                errors.append(f"{where}: duplicate title (also used by '{seen_titles[key]}')")
            seen_titles.setdefault(key, str(slug))
        elif title is not None:
            errors.append(f"{where}: title must be a string")

        category = entry.get("category")
        if category is not None:
            if category not in categories:
                errors.append(f"{where}: category '{category}' is not declared under 'categories'")
            else:
                per_category[category] += 1

        vulnerability_type = entry.get("vulnerability_type")
        if vulnerability_type is not None and not (
            isinstance(vulnerability_type, str) and SNAKE_PATTERN.match(vulnerability_type)
        ):
            errors.append(f"{where}: vulnerability_type must be snake_case")
        if entry.get("stack_trigger") is not None and not _is_text(entry.get("stack_trigger")):
            errors.append(f"{where}: stack_trigger must be a non-empty string")

        flags = entry.get("binary_flags")
        if flags is not None:
            if not isinstance(flags, dict):
                errors.append(f"{where}: binary_flags must be a mapping")
            else:
                for key, value in flags.items():
                    if key not in BINARY_FLAG_KEYS:
                        errors.append(f"{where}: unknown binary flag '{key}'")
                    if not isinstance(value, bool):
                        errors.append(f"{where}: binary flag '{key}' must be true or false")

        for field in PROSE_FIELDS:
            value = entry.get(field)
            if value is None:
                continue
            if not _is_text(value):
                errors.append(f"{where}: {field} must be a non-empty string")
            elif len(value.strip()) < MIN_PROSE_CHARS:
                errors.append(f"{where}: {field} is too short to be useful (< {MIN_PROSE_CHARS} chars)")

        archetypes = entry.get("archetypes")
        if archetypes is not None and not (
            isinstance(archetypes, list) and archetypes and all(a in ARCHETYPES for a in archetypes)
        ):
            errors.append(f"{where}: archetypes must be a non-empty list drawn from {', '.join(ARCHETYPES)}")

        steps = entry.get("verification_steps")
        if steps is not None and not _is_text_list(steps):
            errors.append(f"{where}: verification_steps must be a non-empty list of strings")

        detection = entry.get("detection")
        if detection is not None:
            if not isinstance(detection, list) or not detection:
                errors.append(f"{where}: detection must be a non-empty list of detector specs")
            else:
                kinds = set()
                for d_index, detector in enumerate(detection):
                    _validate_detector(f"{where} detection[{d_index}]", detector, errors, counters)
                    if isinstance(detector, dict) and detector.get("type") in DETECTOR_TYPES:
                        detector_types[detector["type"]] += 1
                        kinds.add(detector["type"])
                if kinds & set(DETERMINISTIC_TYPES):
                    deterministic_entries += 1

        references = entry.get("references", [])
        if references is None:
            references = []
        if not isinstance(references, list):
            errors.append(f"{where}: references must be a list")
        else:
            for r_index, reference in enumerate(references):
                r_where = f"{where} references[{r_index}]"
                if not isinstance(reference, dict) or not _is_text(reference.get("title")):
                    errors.append(f"{r_where}: must be a mapping with a non-empty 'title'")
                    continue
                url = reference.get("url")
                if url is not None and not (isinstance(url, str) and url.startswith("https://")):
                    errors.append(f"{r_where}: url must be null or start with https://")
                if set(reference) - {"title", "url"}:
                    errors.append(f"{r_where}: only 'title' and 'url' are allowed")

    for slug in REQUIRED_SLUGS:
        if slug not in seen_slugs:
            errors.append(f"catalogue: mandatory entry '{slug}' is missing")
    if entries and deterministic_entries < MIN_DETERMINISTIC_ENTRIES:
        errors.append(
            f"catalogue: only {deterministic_entries} entries have a deterministic detector; "
            f"at least {MIN_DETERMINISTIC_ENTRIES} are required"
        )
    for name in categories:
        if entries and per_category[name] == 0:
            errors.append(f"categories: '{name}' is declared but has no entries")

    report = {
        "entries": len(entries),
        "categories": dict(sorted(per_category.items())),
        "detector_types": dict(sorted(detector_types.items())),
        "deterministic_entries": deterministic_entries,
        "llm_only_entries": len(entries) - deterministic_entries,
        "regexes_compiled": counters["regexes_compiled"],
        "regex_samples_checked": counters["regex_samples"],
    }
    return errors, report


def normalized_flags(flags: dict[str, Any] | None) -> dict[str, bool]:
    """Full flag object: every known key present, unspecified ones false."""
    flags = flags or {}
    return {key: bool(flags.get(key, False)) for key in BINARY_FLAG_KEYS}


# --------------------------------------------------------------------------- #
# Novelty check (pure Python; the SQL path mirrors it with pg_trgm)
# --------------------------------------------------------------------------- #
def _norm(*parts: Any) -> str:
    return re.sub(r"\s+", " ", " ".join(str(p or "") for p in parts)).strip().lower()


def short_signature(row: dict[str, Any]) -> str:
    return _norm(row.get("vulnerability_type"), row.get("stack_trigger"))


def full_signature(row: dict[str, Any]) -> str:
    return _norm(row.get("vulnerability_type"), row.get("stack_trigger"), row.get("fix_recommendation"))


def difflib_score(a: dict[str, Any], b: dict[str, Any]) -> float:
    ratio = difflib.SequenceMatcher
    return max(
        ratio(None, short_signature(a), short_signature(b)).ratio(),
        ratio(None, full_signature(a), full_signature(b)).ratio(),
    )


def best_match_difflib(discovery: dict[str, Any], catalog_rows: list[dict[str, Any]]):
    """Return (closest catalogue row or None, similarity 0..1)."""
    best, best_score = None, 0.0
    for row in catalog_rows:
        score = difflib_score(discovery, row)
        if score > best_score:
            best, best_score = row, score
    return best, best_score


def promoted_slug(discovery: dict[str, Any]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", str(discovery.get("vulnerability_type", "")).lower()).strip("-")
    return f"disc-{discovery['id']}-{base[:50].strip('-') or 'discovery'}"


def promoted_title(discovery: dict[str, Any]) -> str:
    kind = str(discovery.get("vulnerability_type", "")).replace("_", " ").strip().capitalize()
    trigger = str(discovery.get("stack_trigger", "")).strip()
    return f"{kind} ({trigger[:90]})" if trigger else kind


# --------------------------------------------------------------------------- #
# Database operations
# --------------------------------------------------------------------------- #
def _db():
    from repo_rover import db_tools  # imported lazily: validate/stats stay offline

    return db_tools


SEED_COLUMNS = (
    "slug", "title", "category", "vulnerability_type", "stack_trigger",
    "binary_flags", "symptom", "why_ai_makes_this_mistake", "detection",
    "fix_recommendation", "verification_steps", "reference_links", "archetypes",
)


def seed_rows(data: dict[str, Any]) -> list[tuple]:
    from psycopg2.extras import Json

    rows = []
    for entry in data["entries"]:
        rows.append(
            (
                entry["slug"],
                entry["title"].strip(),
                entry["category"],
                entry["vulnerability_type"],
                entry["stack_trigger"].strip(),
                Json(normalized_flags(entry["binary_flags"])),
                entry["symptom"].strip(),
                entry["why_ai_makes_this_mistake"].strip(),
                Json(entry["detection"]),
                entry["fix_recommendation"].strip(),
                Json(entry["verification_steps"]),
                Json(entry.get("references") or []),
                Json(entry.get("archetypes") or []),
            )
        )
    return rows


def seed_catalog(data: dict[str, Any], conn=None) -> dict[str, int]:
    """Upsert every YAML entry into known_failure_nodes by slug. Idempotent."""
    from psycopg2.extras import execute_values

    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in SEED_COLUMNS if c != "slug")
    sql = f"""
        INSERT INTO known_failure_nodes ({", ".join(SEED_COLUMNS)}, source, updated_at)
        VALUES %s
        ON CONFLICT (slug) DO UPDATE SET {updates}, source = 'seed', updated_at = now()
        RETURNING (xmax = 0) AS inserted
    """
    template = "(" + ", ".join(["%s"] * len(SEED_COLUMNS)) + ", 'seed', now())"

    def run(connection) -> dict[str, int]:
        with connection.cursor() as cur:
            results = execute_values(cur, sql, seed_rows(data), template=template, fetch=True, page_size=500)
        inserted = sum(1 for (was_inserted,) in results if was_inserted)
        return {"inserted": inserted, "updated": len(results) - inserted, "total": len(results)}

    if conn is not None:
        return run(conn)
    with _db()._connection() as connection:
        return run(connection)


def db_stats() -> list[tuple[str, str, int]]:
    with _db()._connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(category, '(uncategorised)'), source, count(*)
                FROM known_failure_nodes
                GROUP BY 1, 2
                ORDER BY 1, 2
                """
            )
            return [(c, s, n) for c, s, n in cur.fetchall()]


def trgm_available(cur) -> bool:
    cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'")
    return cur.fetchone() is not None


def _best_match_trgm(cur, discovery: dict[str, Any]):
    cur.execute(
        """
        SELECT id, slug, vulnerability_type, stack_trigger,
               GREATEST(
                   similarity(lower(vulnerability_type || ' ' || stack_trigger), %s),
                   similarity(lower(vulnerability_type || ' ' || stack_trigger || ' ' || fix_recommendation), %s)
               ) AS score
        FROM known_failure_nodes
        ORDER BY score DESC, id
        LIMIT 1
        """,
        (short_signature(discovery), full_signature(discovery)),
    )
    row = cur.fetchone()
    if row is None:
        return None, 0.0
    match = {"id": row[0], "slug": row[1], "vulnerability_type": row[2], "stack_trigger": row[3]}
    return match, float(row[4] or 0.0)


def promote_discoveries(
    conn,
    ids: list[int] | None = None,
    threshold: float | None = None,
    category: str = PROMOTED_CATEGORY,
    dry_run: bool = False,
    force_difflib: bool = False,
) -> dict[str, Any]:
    """Promote pending discoveries that are novel; mark the rest as duplicates.

    Novelty is the best similarity between the discovery and any catalogue
    entry: pg_trgm `similarity()` when the extension is installed, otherwise
    Python difflib. At or above the threshold the discovery is a duplicate.
    The caller owns the transaction (commit or roll back).
    """
    from psycopg2.extras import Json, RealDictCursor

    with conn.cursor() as cur:
        use_trgm = (not force_difflib) and trgm_available(cur)
    method = "pg_trgm" if use_trgm else "difflib"
    limit = threshold if threshold is not None else (TRGM_THRESHOLD if use_trgm else DIFFLIB_THRESHOLD)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        query = """
            SELECT id, vulnerability_type, stack_trigger, binary_flags,
                   fix_recommendation, source_repo, confidence, evidence
            FROM dynamic_discoveries
            WHERE review_status = 'pending'
        """
        if ids:
            cur.execute(query + " AND id = ANY(%s) ORDER BY id", (list(ids),))
        else:
            cur.execute(query + " ORDER BY id")
        discoveries = [dict(row) for row in cur.fetchall()]

        catalog_rows: list[dict[str, Any]] = []
        if not use_trgm or dry_run:
            cur.execute(
                "SELECT id, slug, vulnerability_type, stack_trigger, fix_recommendation "
                "FROM known_failure_nodes"
            )
            catalog_rows = [dict(row) for row in cur.fetchall()]

    outcomes = []
    with conn.cursor() as cur:
        for discovery in discoveries:
            if use_trgm and not dry_run:
                match, score = _best_match_trgm(cur, discovery)
            elif use_trgm:
                # Dry run: earlier would-be promotions are not in the table, so
                # compare against them in memory as well.
                match, score = _best_match_trgm(cur, discovery)
                pending = [r for r in catalog_rows if r.get("id") is None]
                local, local_score = best_match_difflib(discovery, pending)
                if local_score > score:
                    match, score = local, local_score
            else:
                match, score = best_match_difflib(discovery, catalog_rows)

            outcome = {
                "discovery_id": discovery["id"],
                "vulnerability_type": discovery["vulnerability_type"],
                "similarity": round(score, 3),
                "closest": match["slug"] if match else None,
            }
            if match is not None and score >= limit:
                outcome["action"] = "duplicate"
                if not dry_run:
                    cur.execute(
                        "UPDATE dynamic_discoveries SET review_status = 'duplicate', "
                        "catalog_id = %s, similarity = %s, reviewed_at = now() WHERE id = %s",
                        (match.get("id"), score, discovery["id"]),
                    )
            else:
                slug = promoted_slug(discovery)
                outcome["action"] = "promoted"
                outcome["slug"] = slug
                new_row = {
                    "id": None,
                    "slug": slug,
                    "vulnerability_type": discovery["vulnerability_type"],
                    "stack_trigger": discovery["stack_trigger"],
                    "fix_recommendation": discovery["fix_recommendation"],
                }
                if not dry_run:
                    cur.execute(
                        """
                        INSERT INTO known_failure_nodes
                            (slug, title, category, vulnerability_type, stack_trigger,
                             binary_flags, fix_recommendation, detection, source,
                             promoted_from_discovery_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'promoted', %s)
                        ON CONFLICT (slug) DO UPDATE SET updated_at = now()
                        RETURNING id
                        """,
                        (
                            slug,
                            promoted_title(discovery),
                            category,
                            discovery["vulnerability_type"],
                            discovery["stack_trigger"],
                            Json(normalized_flags(discovery.get("binary_flags"))),
                            discovery["fix_recommendation"],
                            Json(
                                [
                                    {
                                        "type": "llm_only",
                                        "question": "Does this repository exhibit: "
                                        + promoted_title(discovery)
                                        + "?",
                                    }
                                ]
                            ),
                            discovery["id"],
                        ),
                    )
                    new_row["id"] = cur.fetchone()[0]
                    cur.execute(
                        "UPDATE dynamic_discoveries SET review_status = 'promoted', "
                        "catalog_id = %s, similarity = %s, reviewed_at = now() WHERE id = %s",
                        (new_row["id"], score, discovery["id"]),
                    )
                catalog_rows.append(new_row)
            outcomes.append(outcome)

    return {
        "method": method,
        "threshold": limit,
        "dry_run": dry_run,
        "promoted": sum(1 for o in outcomes if o["action"] == "promoted"),
        "duplicates": sum(1 for o in outcomes if o["action"] == "duplicate"),
        "outcomes": outcomes,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_or_exit(path: Path) -> dict[str, Any]:
    try:
        return load_catalog(path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        sys.exit(f"Could not load {path}: {exc}")


def _print_report(report: dict[str, Any]) -> None:
    print(f"Entries:                 {report['entries']}")
    print(f"  with deterministic detector: {report['deterministic_entries']}")
    print(f"  llm_only:                    {report['llm_only_entries']}")
    print("Detectors by type:       " + ", ".join(f"{k}={v}" for k, v in report["detector_types"].items()))
    print(f"Regexes compiled:        {report['regexes_compiled']}")
    print(f"Regex samples checked:   {report['regex_samples_checked']}")


def cmd_validate(args: argparse.Namespace) -> int:
    path = catalog_path(args.file)
    errors, report = validate_catalog(_load_or_exit(path))
    print(f"Catalogue: {path}")
    _print_report(report)
    if errors:
        print(f"\nINVALID - {len(errors)} problem(s):")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("\nVALID - schema, unique slugs, required fields and detector regexes all pass.")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    if args.db:
        rows = db_stats()
        total = sum(count for _, _, count in rows)
        print(f"{'category':<28} {'source':<10} {'entries':>7}")
        for category, source, count in rows:
            print(f"{category:<28} {source:<10} {count:>7}")
        print(f"{'TOTAL':<28} {'':<10} {total:>7}")
        return 0
    path = catalog_path(args.file)
    _, report = validate_catalog(_load_or_exit(path))
    print(f"Catalogue: {path}")
    print(f"{'category':<28} {'entries':>7}")
    for category, count in report["categories"].items():
        print(f"{category:<28} {count:>7}")
    print(f"{'TOTAL':<28} {report['entries']:>7}")
    print()
    _print_report(report)
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    path = catalog_path(args.file)
    data = _load_or_exit(path)
    errors, _ = validate_catalog(data)
    if errors:
        print(f"Refusing to seed: {path} has {len(errors)} validation problem(s).")
        print("Run `reporover catalog validate` for details.")
        return 1
    result = seed_catalog(data)
    print(
        f"Seeded {result['total']} entries into known_failure_nodes: "
        f"{result['inserted']} inserted, {result['updated']} updated."
    )
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    with _db()._connection() as conn:
        result = promote_discoveries(
            conn,
            ids=args.id,
            threshold=args.threshold,
            category=args.category,
            dry_run=args.dry_run,
            force_difflib=args.difflib,
        )
    mode = " (dry run - nothing written)" if result["dry_run"] else ""
    print(f"Novelty check: {result['method']}, threshold {result['threshold']}{mode}")
    if not result["outcomes"]:
        print("No pending discoveries to review.")
        return 0
    for outcome in result["outcomes"]:
        target = outcome.get("slug") or outcome["closest"]
        print(
            f"  discovery {outcome['discovery_id']:>5}  {outcome['action']:<9} "
            f"similarity {outcome['similarity']:.3f}  "
            f"{'->' if outcome['action'] == 'promoted' else 'of'} {target}"
        )
    print(f"Promoted {result['promoted']}, duplicates {result['duplicates']}.")
    return 0


def add_catalog_commands(subparsers) -> None:
    catalog = subparsers.add_parser("catalog", help="Manage the failure catalogue")
    commands = catalog.add_subparsers(dest="catalog_command", required=True)

    validate = commands.add_parser("validate", help="Check the YAML seed file (offline)")
    validate.add_argument("--file", help="Path to a catalogue YAML file")
    validate.set_defaults(func=cmd_validate)

    stats = commands.add_parser("stats", help="Print entry counts per category")
    stats.add_argument("--file", help="Path to a catalogue YAML file")
    stats.add_argument("--db", action="store_true", help="Count rows in Postgres instead of the YAML file")
    stats.set_defaults(func=cmd_stats)

    seed = commands.add_parser("seed", help="Upsert the YAML seed file into Postgres by slug")
    seed.add_argument("--file", help="Path to a catalogue YAML file")
    seed.set_defaults(func=cmd_seed)

    promote = commands.add_parser(
        "promote", help="Promote novel staged discoveries into the catalogue"
    )
    promote.add_argument("--id", type=int, action="append", help="Discovery id (repeatable); default: all pending")
    promote.add_argument("--threshold", type=float, help="Similarity at or above which a discovery is a duplicate")
    promote.add_argument("--category", default=PROMOTED_CATEGORY, help="Category for promoted entries")
    promote.add_argument("--dry-run", action="store_true", help="Show what would happen without writing")
    promote.add_argument("--difflib", action="store_true", help="Force the Python difflib novelty check")
    promote.set_defaults(func=cmd_promote)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reporover")
    add_catalog_commands(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(["catalog", *sys.argv[1:]]))
