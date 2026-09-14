CREATE SCHEMA IF NOT EXISTS reconciliation;

-- One row per issue (a transaction may have multiple issues)
CREATE TABLE IF NOT EXISTS reconciliation.reconciliation_issues (
    issue_id            BIGSERIAL PRIMARY KEY,
    batch_id            TEXT NOT NULL,
    transaction_id      TEXT NOT NULL,
    settlement_id       TEXT,
    issue_type          TEXT NOT NULL,   -- MISSING_SETTLEMENT, AMOUNT_MISMATCH, DUPLICATE_TXN, DUPLICATE_SETTLEMENT, MISSING_FX_RATE, ORPHAN_SETTLEMENT, DELAYED_SETTLEMENT, INVALID_RECORD
    amount_difference   NUMERIC(18,2),
    expected_amount     NUMERIC(18,2),
    actual_amount       NUMERIC(18,2),
    currency            TEXT,
    details             JSONB,
    detected_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at         TIMESTAMPTZ,
    resolution_status   TEXT DEFAULT 'OPEN'  -- OPEN, RESOLVED, WAIVED
);

CREATE INDEX IF NOT EXISTS idx_ri_txn_id ON reconciliation.reconciliation_issues(transaction_id);
CREATE INDEX IF NOT EXISTS idx_ri_batch ON reconciliation.reconciliation_issues(batch_id);
CREATE INDEX IF NOT EXISTS idx_ri_type ON reconciliation.reconciliation_issues(issue_type);
CREATE INDEX IF NOT EXISTS idx_ri_status ON reconciliation.reconciliation_issues(resolution_status);

CREATE TABLE IF NOT EXISTS reconciliation.summary (
    summary_id          BIGSERIAL PRIMARY KEY,
    batch_id            TEXT NOT NULL UNIQUE,
    total_transactions  BIGINT,
    matched             BIGINT,
    missing_settlement  BIGINT,
    amount_mismatch     BIGINT,
    duplicate_txn       BIGINT,
    duplicate_settlement BIGINT,
    missing_fx_rate     BIGINT,
    orphan_settlement   BIGINT,
    delayed_settlement  BIGINT,
    invalid_record      BIGINT,
    precision_score     NUMERIC(6,4),
    recall_score        NUMERIC(6,4),
    computed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
