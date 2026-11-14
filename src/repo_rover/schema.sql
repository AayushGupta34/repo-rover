-- RepoRover schema for Neon Serverless Postgres.
-- Apply with:  uv run python -m repo_rover.db_tools
-- (or paste into the Neon SQL editor). Safe to run repeatedly.

-- Curated, trusted failure knowledge. Read by the Inspector agent.
CREATE TABLE IF NOT EXISTS known_failure_nodes (
    id                 BIGSERIAL PRIMARY KEY,
    vulnerability_type TEXT  NOT NULL,
    stack_trigger      TEXT  NOT NULL,
    binary_flags       JSONB NOT NULL DEFAULT '{}'::jsonb,
    fix_recommendation TEXT  NOT NULL
);

-- Staging table. Written by the Red Teamer and the Market Intelligence
-- Scraper; rows are reviewed before being promoted to known_failure_nodes.
CREATE TABLE IF NOT EXISTS dynamic_discoveries (
    id                 BIGSERIAL PRIMARY KEY,
    vulnerability_type TEXT  NOT NULL,
    stack_trigger      TEXT  NOT NULL,
    binary_flags       JSONB NOT NULL DEFAULT '{}'::jsonb,
    fix_recommendation TEXT  NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_known_failure_nodes_stack_trigger
    ON known_failure_nodes (lower(stack_trigger));

CREATE INDEX IF NOT EXISTS idx_dynamic_discoveries_stack_trigger
    ON dynamic_discoveries (lower(stack_trigger));

CREATE INDEX IF NOT EXISTS idx_known_failure_nodes_binary_flags
    ON known_failure_nodes USING GIN (binary_flags);

CREATE INDEX IF NOT EXISTS idx_dynamic_discoveries_binary_flags
    ON dynamic_discoveries USING GIN (binary_flags);

-- ---------------------------------------------------------------------------
-- Migration 002 (Phase 1): analysis run tracking. Additive and idempotent.
-- ---------------------------------------------------------------------------

-- One row per analyzer run. ledger_summary holds the read ledger: files read,
-- line ranges served and content hashes.
CREATE TABLE IF NOT EXISTS analysis_runs (
    id               BIGSERIAL PRIMARY KEY,
    run_uuid         TEXT        NOT NULL UNIQUE,
    started_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at      TIMESTAMPTZ,
    target_repo      TEXT        NOT NULL,
    status           TEXT        NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'degraded', 'failed')),
    degraded_reasons JSONB       NOT NULL DEFAULT '[]'::jsonb,
    ledger_summary   JSONB       NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_analysis_runs_status
    ON analysis_runs (status);

CREATE INDEX IF NOT EXISTS idx_analysis_runs_started_at
    ON analysis_runs (started_at DESC);

-- Which run produced a staged discovery. NULL for rows written before this
-- migration and for rows from the meta crew, which has no analysis run.
ALTER TABLE dynamic_discoveries
    ADD COLUMN IF NOT EXISTS run_id BIGINT;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'fk_dynamic_discoveries_run_id'
          AND conrelid = 'dynamic_discoveries'::regclass
    ) THEN
        ALTER TABLE dynamic_discoveries
            ADD CONSTRAINT fk_dynamic_discoveries_run_id
            FOREIGN KEY (run_id) REFERENCES analysis_runs (id)
            ON DELETE SET NULL;
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS idx_dynamic_discoveries_run_id
    ON dynamic_discoveries (run_id);

-- ---------------------------------------------------------------------------
-- Migration 003 (Phase 2): the failure catalogue. Additive and idempotent.
-- ---------------------------------------------------------------------------

-- known_failure_nodes becomes the catalogue. The original five columns keep
-- their meaning, so QueryDatabaseTool and existing rows continue to work.
-- "references" is a reserved word in SQL, hence reference_links.
ALTER TABLE known_failure_nodes
    ADD COLUMN IF NOT EXISTS slug                       TEXT,
    ADD COLUMN IF NOT EXISTS title                      TEXT,
    ADD COLUMN IF NOT EXISTS category                   TEXT,
    ADD COLUMN IF NOT EXISTS symptom                    TEXT,
    ADD COLUMN IF NOT EXISTS why_ai_makes_this_mistake  TEXT,
    ADD COLUMN IF NOT EXISTS detection                  JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS verification_steps         JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS reference_links            JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS source                     TEXT  NOT NULL DEFAULT 'manual',
    ADD COLUMN IF NOT EXISTS promoted_from_discovery_id BIGINT,
    ADD COLUMN IF NOT EXISTS created_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    ADD COLUMN IF NOT EXISTS updated_at                 TIMESTAMPTZ NOT NULL DEFAULT now();

-- Seed upserts key on slug. NULL slugs (legacy rows) do not collide.
CREATE UNIQUE INDEX IF NOT EXISTS uq_known_failure_nodes_slug
    ON known_failure_nodes (slug);

CREATE INDEX IF NOT EXISTS idx_known_failure_nodes_category
    ON known_failure_nodes (category);

-- Provenance and review state for staged discoveries.
ALTER TABLE dynamic_discoveries
    ADD COLUMN IF NOT EXISTS source_repo   TEXT,
    ADD COLUMN IF NOT EXISTS confidence    REAL,
    ADD COLUMN IF NOT EXISTS evidence      JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS review_status TEXT  NOT NULL DEFAULT 'pending',
    ADD COLUMN IF NOT EXISTS catalog_id    BIGINT,
    ADD COLUMN IF NOT EXISTS similarity    REAL,
    ADD COLUMN IF NOT EXISTS reviewed_at   TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_dynamic_discoveries_review_status
    ON dynamic_discoveries (review_status);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_known_failure_nodes_source') THEN
        ALTER TABLE known_failure_nodes
            ADD CONSTRAINT ck_known_failure_nodes_source
            CHECK (source IN ('manual', 'seed', 'promoted'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_known_failure_nodes_discovery') THEN
        ALTER TABLE known_failure_nodes
            ADD CONSTRAINT fk_known_failure_nodes_discovery
            FOREIGN KEY (promoted_from_discovery_id) REFERENCES dynamic_discoveries (id)
            ON DELETE SET NULL;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_dynamic_discoveries_confidence') THEN
        ALTER TABLE dynamic_discoveries
            ADD CONSTRAINT ck_dynamic_discoveries_confidence
            CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_dynamic_discoveries_review_status') THEN
        ALTER TABLE dynamic_discoveries
            ADD CONSTRAINT ck_dynamic_discoveries_review_status
            CHECK (review_status IN ('pending', 'promoted', 'duplicate', 'rejected'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_dynamic_discoveries_catalog_id') THEN
        ALTER TABLE dynamic_discoveries
            ADD CONSTRAINT fk_dynamic_discoveries_catalog_id
            FOREIGN KEY (catalog_id) REFERENCES known_failure_nodes (id)
            ON DELETE SET NULL;
    END IF;
END
$$;

-- Trigram similarity powers the novelty check in `reporover catalog promote`.
-- If the extension cannot be installed, promotion falls back to Python difflib.
DO $$
BEGIN
    BEGIN
        CREATE EXTENSION IF NOT EXISTS pg_trgm;
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'pg_trgm is unavailable: %', SQLERRM;
    END;
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm') THEN
        EXECUTE 'CREATE INDEX IF NOT EXISTS idx_known_failure_nodes_trgm '
             || 'ON known_failure_nodes USING gin '
             || '((lower(vulnerability_type || '' '' || stack_trigger)) gin_trgm_ops)';
    END IF;
END
$$;

-- ---------------------------------------------------------------------------
-- Migration 004 (Phase 4): idempotent discovery writes. Additive and idempotent.
-- ---------------------------------------------------------------------------

-- A stable hash of (source repo, vulnerability type, stack, fix). Inserts use
-- ON CONFLICT (dedupe_key) DO NOTHING, so a write that is retried after a lost
-- acknowledgement, or repeated by a resumed run, cannot create a second row.
-- NULL for rows written before this migration; NULLs never conflict.
ALTER TABLE dynamic_discoveries
    ADD COLUMN IF NOT EXISTS dedupe_key TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS uq_dynamic_discoveries_dedupe_key
    ON dynamic_discoveries (dedupe_key);

-- ---------------------------------------------------------------------------
-- Migration 005 (Phase 6): archetype gating for catalogue rules. Additive.
-- ---------------------------------------------------------------------------

-- Repository archetypes (backend, frontend, static_site, library) a rule's
-- absence and manifest detectors apply to. Empty means every archetype.
ALTER TABLE known_failure_nodes
    ADD COLUMN IF NOT EXISTS archetypes JSONB NOT NULL DEFAULT '[]'::jsonb;
