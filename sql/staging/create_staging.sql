-- Staging tables: receive raw COPY loads before upsert
CREATE SCHEMA IF NOT EXISTS staging;

CREATE TABLE IF NOT EXISTS staging.transactions (
    batch_id            TEXT NOT NULL,
    step                INTEGER,
    type                TEXT,
    amount              NUMERIC(18,2),
    name_orig           TEXT,
    old_balance_orig    NUMERIC(18,2),
    new_balance_orig    NUMERIC(18,2),
    name_dest           TEXT,
    old_balance_dest    NUMERIC(18,2),
    new_balance_dest    NUMERIC(18,2),
    is_fraud            SMALLINT,
    is_flagged_fraud    SMALLINT,
    -- Enriched fields (added by M2 generator)
    transaction_id      TEXT,
    currency            TEXT,
    fx_pair             TEXT,
    transaction_timestamp TIMESTAMPTZ,
    ingestion_timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS staging.settlements (
    batch_id            TEXT NOT NULL,
    settlement_id       TEXT,
    transaction_id      TEXT,
    settled_amount      NUMERIC(18,2),
    settlement_currency TEXT,
    settlement_timestamp TIMESTAMPTZ,
    status              TEXT,
    ingestion_timestamp TIMESTAMPTZ DEFAULT NOW()
);
