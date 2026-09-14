CREATE SCHEMA IF NOT EXISTS metadata;

CREATE TABLE IF NOT EXISTS metadata.pipeline_batches (
    batch_id            TEXT PRIMARY KEY,
    source_file         TEXT,
    status              TEXT NOT NULL DEFAULT 'PENDING',  -- PENDING, RUNNING, COMPLETED, FAILED
    rows_ingested       BIGINT,
    rows_valid          BIGINT,
    rows_rejected       BIGINT,
    rows_loaded_warehouse BIGINT,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMPTZ,
    error_message       TEXT,
    retry_count         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS metadata.quarantine (
    quarantine_id       BIGSERIAL PRIMARY KEY,
    batch_id            TEXT NOT NULL,
    source_file         TEXT NOT NULL,
    record_type         TEXT NOT NULL,  -- TRANSACTION, SETTLEMENT
    payload             JSONB NOT NULL,
    reasons             TEXT[] NOT NULL,
    quarantined_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reprocess_flag      BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_quar_batch ON metadata.quarantine(batch_id);
CREATE INDEX IF NOT EXISTS idx_quar_reprocess ON metadata.quarantine(reprocess_flag) WHERE reprocess_flag = TRUE;
