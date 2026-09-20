"""
loader.py — Bulk load to PostgreSQL using COPY ... FROM STDIN then upsert.

Rules:
- COPY to staging table first (fast bulk insert)
- INSERT INTO warehouse ... ON CONFLICT DO UPDATE (idempotent upsert)
- NEVER row-by-row inserts
- NEVER load 6.3M rows into pandas — stream from parquet using pyarrow batch reader
- Money comes in as Decimal strings — cast to NUMERIC(18,2) in SQL
"""
from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def _iter_parquet_batches(
    parquet_path: Path,
    batch_size: int = 10_000,
) -> Iterator[pa.RecordBatch]:
    """Yield pyarrow RecordBatches from a parquet file/directory. Never loads all at once."""
    if parquet_path.is_dir():
        # Partitioned parquet: read all part files
        dataset = pq.ParquetDataset(str(parquet_path))
        for fragment in dataset.fragments:
            reader = fragment.open()
            for batch in reader.iter_batches(batch_size=batch_size):
                yield batch
    else:
        parquet_file = pq.ParquetFile(str(parquet_path))
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            yield batch


def _decimal_to_str(val) -> str | None:
    """Convert Decimal or pyarrow Decimal to string for COPY. None stays None."""
    if val is None:
        return None
    return str(val)


def copy_transactions_to_staging(
    conn: psycopg.Connection,
    parquet_path: Path,
    batch_id: str,
) -> int:
    """
    COPY transactions from parquet to staging.transactions.
    Returns row count copied.
    """
    logger.info("COPY transactions to staging — batch=%s, source=%s", batch_id, parquet_path)
    total = 0

    # Columns in staging.transactions (must match CREATE TABLE order)
    staging_cols = [
        "batch_id", "step", "type", "amount", "name_orig",
        "old_balance_orig", "new_balance_orig", "name_dest",
        "old_balance_dest", "new_balance_dest", "is_fraud",
        "is_flagged_fraud", "transaction_id", "currency",
        "fx_pair", "transaction_timestamp",
    ]

    copy_sql = (
        f"COPY staging.transactions ({', '.join(staging_cols)}) "
        "FROM STDIN WITH (FORMAT TEXT, DELIMITER E'\\t', NULL '')"
    )

    with conn.cursor() as cur:
        for batch in _iter_parquet_batches(parquet_path):
            buf = io.StringIO()
            for i in range(batch.num_rows):
                row = {col: batch.column(col)[i].as_py() if col in batch.schema.names else None
                       for col in staging_cols}

                line = "\t".join(
                    "" if row.get(col) is None else str(row[col])
                    for col in staging_cols
                )
                buf.write(line + "\n")
            buf.seek(0)
            with cur.copy(copy_sql) as copy_obj:
                copy_obj.write(buf.read())
            total += batch.num_rows
            logger.debug("Copied batch of %d rows (total so far: %d)", batch.num_rows, total)

    logger.info("COPY complete — %d rows to staging.transactions", total)
    return total


def upsert_transactions_to_warehouse(conn: psycopg.Connection, batch_id: str) -> int:
    """
    INSERT INTO warehouse.transactions ... ON CONFLICT DO UPDATE
    from staging.transactions for this batch_id.
    Returns rows upserted.
    """
    upsert_sql = """
    INSERT INTO warehouse.transactions (
        transaction_id, batch_id, step, type, amount,
        name_orig, old_balance_orig, new_balance_orig,
        name_dest, old_balance_dest, new_balance_dest,
        is_fraud, is_flagged_fraud, currency, fx_pair,
        transaction_timestamp, ingestion_timestamp
    )
    SELECT
        transaction_id,
        batch_id,
        step,
        type,
        amount::NUMERIC(18,2),
        name_orig,
        old_balance_orig::NUMERIC(18,2),
        new_balance_orig::NUMERIC(18,2),
        name_dest,
        old_balance_dest::NUMERIC(18,2),
        new_balance_dest::NUMERIC(18,2),
        is_fraud,
        is_flagged_fraud,
        currency,
        fx_pair,
        transaction_timestamp,
        ingestion_timestamp
    FROM staging.transactions
    WHERE batch_id = %(batch_id)s
      AND transaction_id IS NOT NULL
    ON CONFLICT (transaction_id) DO UPDATE
        SET
            batch_id = EXCLUDED.batch_id,
            amount = EXCLUDED.amount,
            fx_pair = EXCLUDED.fx_pair,
            currency = EXCLUDED.currency,
            ingestion_timestamp = EXCLUDED.ingestion_timestamp
    """
    with conn.cursor() as cur:
        cur.execute(upsert_sql, {"batch_id": batch_id})
        count = cur.rowcount
    logger.info("Upserted %d rows to warehouse.transactions for batch=%s", count, batch_id)
    return count


def copy_settlements_to_staging(
    conn: psycopg.Connection,
    parquet_path: Path,
    batch_id: str,
) -> int:
    """COPY settlements from parquet to staging.settlements."""
    logger.info("COPY settlements to staging — batch=%s", batch_id)
    total = 0

    staging_cols = [
        "batch_id", "settlement_id", "transaction_id",
        "settled_amount", "settlement_currency", "settlement_timestamp", "status",
    ]

    copy_sql = (
        f"COPY staging.settlements ({', '.join(staging_cols)}) "
        "FROM STDIN WITH (FORMAT TEXT, DELIMITER E'\\t', NULL '')"
    )

    with conn.cursor() as cur:
        for batch in _iter_parquet_batches(parquet_path):
            buf = io.StringIO()
            for i in range(batch.num_rows):
                row = {col: batch.column(col)[i].as_py() if col in batch.schema.names else None
                       for col in staging_cols}
                line = "\t".join(
                    "" if row.get(col) is None else str(row[col])
                    for col in staging_cols
                )
                buf.write(line + "\n")
            buf.seek(0)
            with cur.copy(copy_sql) as copy_obj:
                copy_obj.write(buf.read())
            total += batch.num_rows

    logger.info("COPY complete — %d rows to staging.settlements", total)
    return total


def upsert_settlements_to_warehouse(conn: psycopg.Connection, batch_id: str) -> int:
    """Upsert settlements from staging to warehouse."""
    upsert_sql = """
    INSERT INTO warehouse.settlements (
        settlement_id, transaction_id, batch_id,
        settled_amount, settlement_currency,
        settlement_timestamp, status, ingestion_timestamp
    )
    SELECT
        settlement_id,
        transaction_id,
        batch_id,
        settled_amount::NUMERIC(18,2),
        settlement_currency,
        settlement_timestamp,
        status,
        ingestion_timestamp
    FROM staging.settlements
    WHERE batch_id = %(batch_id)s
      AND settlement_id IS NOT NULL
    ON CONFLICT (settlement_id) DO UPDATE
        SET
            settled_amount = EXCLUDED.settled_amount,
            status = EXCLUDED.status,
            ingestion_timestamp = EXCLUDED.ingestion_timestamp
    """
    with conn.cursor() as cur:
        cur.execute(upsert_sql, {"batch_id": batch_id})
        count = cur.rowcount
    logger.info("Upserted %d rows to warehouse.settlements for batch=%s", count, batch_id)
    return count


def load_reconciliation_issues(
    conn: psycopg.Connection,
    parquet_path: Path,
    batch_id: str,
) -> int:
    """INSERT reconciliation issues (no upsert — issues are append-only)."""
    logger.info("Loading reconciliation issues — batch=%s, source=%s", batch_id, parquet_path)

    insert_sql = """
    INSERT INTO reconciliation.reconciliation_issues
        (batch_id, transaction_id, settlement_id, issue_type,
         amount_difference, expected_amount, actual_amount,
         currency, details, detected_at)
    VALUES
        (%(batch_id)s, %(transaction_id)s, %(settlement_id)s, %(issue_type)s,
         %(amount_difference)s, %(expected_amount)s, %(actual_amount)s,
         %(currency)s, %(details)s, %(detected_at)s)
    """
    total = 0

    with conn.cursor() as cur:
        for batch in _iter_parquet_batches(parquet_path):
            rows = []
            for i in range(batch.num_rows):
                rows.append({
                    "batch_id": batch_id,
                    "transaction_id": batch.column("transaction_id")[i].as_py(),
                    "settlement_id": batch.column("settlement_id")[i].as_py() if "settlement_id" in batch.schema.names else None,
                    "issue_type": batch.column("issue_type")[i].as_py(),
                    "amount_difference": str(batch.column("amount_difference")[i].as_py()) if batch.column("amount_difference")[i].as_py() is not None else None,
                    "expected_amount": str(batch.column("expected_amount")[i].as_py()) if batch.column("expected_amount")[i].as_py() is not None else None,
                    "actual_amount": str(batch.column("actual_amount")[i].as_py()) if batch.column("actual_amount")[i].as_py() is not None else None,
                    "currency": batch.column("currency")[i].as_py() if "currency" in batch.schema.names else None,
                    "details": batch.column("details")[i].as_py() if "details" in batch.schema.names else None,
                    "detected_at": batch.column("detected_at")[i].as_py() if "detected_at" in batch.schema.names else None,
                })
            cur.executemany(insert_sql, rows)
            total += len(rows)

    logger.info("Loaded %d reconciliation issues for batch=%s", total, batch_id)
    return total

