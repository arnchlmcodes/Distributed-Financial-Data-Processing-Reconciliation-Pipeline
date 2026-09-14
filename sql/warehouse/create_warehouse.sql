CREATE SCHEMA IF NOT EXISTS warehouse;

CREATE TABLE IF NOT EXISTS warehouse.transactions (
    transaction_id          TEXT PRIMARY KEY,
    batch_id                TEXT NOT NULL,
    step                    INTEGER NOT NULL,
    type                    TEXT NOT NULL,
    amount                  NUMERIC(18,2) NOT NULL,
    name_orig               TEXT NOT NULL,
    old_balance_orig        NUMERIC(18,2),
    new_balance_orig        NUMERIC(18,2),
    name_dest               TEXT NOT NULL,
    old_balance_dest        NUMERIC(18,2),
    new_balance_dest        NUMERIC(18,2),
    is_fraud                SMALLINT NOT NULL DEFAULT 0,
    is_flagged_fraud        SMALLINT NOT NULL DEFAULT 0,
    currency                TEXT NOT NULL,
    fx_pair                 TEXT,
    fx_rate                 NUMERIC(18,6),
    amount_usd              NUMERIC(18,2),
    transaction_timestamp   TIMESTAMPTZ NOT NULL,
    ingestion_timestamp     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_wh_txn_timestamp ON warehouse.transactions(transaction_timestamp);
CREATE INDEX IF NOT EXISTS idx_wh_txn_type ON warehouse.transactions(type);
CREATE INDEX IF NOT EXISTS idx_wh_txn_batch ON warehouse.transactions(batch_id);
CREATE INDEX IF NOT EXISTS idx_wh_txn_name_orig ON warehouse.transactions(name_orig);

CREATE TABLE IF NOT EXISTS warehouse.settlements (
    settlement_id           TEXT PRIMARY KEY,
    transaction_id          TEXT NOT NULL,
    batch_id                TEXT NOT NULL,
    settled_amount          NUMERIC(18,2) NOT NULL,
    settlement_currency     TEXT NOT NULL,
    settlement_timestamp    TIMESTAMPTZ,
    status                  TEXT NOT NULL,
    ingestion_timestamp     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_settlement_txn FOREIGN KEY (transaction_id)
        REFERENCES warehouse.transactions(transaction_id)
        ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX IF NOT EXISTS idx_wh_stl_txn_id ON warehouse.settlements(transaction_id);
CREATE INDEX IF NOT EXISTS idx_wh_stl_batch ON warehouse.settlements(batch_id);

CREATE TABLE IF NOT EXISTS warehouse.fx_rates (
    fx_rate_id          SERIAL PRIMARY KEY,
    currency_pair       TEXT NOT NULL,
    rate                NUMERIC(18,6) NOT NULL,
    effective_timestamp TIMESTAMPTZ NOT NULL,
    batch_id            TEXT,
    UNIQUE (currency_pair, effective_timestamp)
);

CREATE INDEX IF NOT EXISTS idx_fx_pair_ts ON warehouse.fx_rates(currency_pair, effective_timestamp);
