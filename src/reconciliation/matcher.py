"""
matcher.py — Reconciliation matcher: joins warehouse tables and produces issue rows.

Runs SQL against PostgreSQL (not Spark) for final reconciliation.
Uses chunked cursors to avoid loading all rows into memory.
"""
from __future__ import annotations

import logging

import psycopg

from src.config import settings
from src.database.connection import managed_connection

logger = logging.getLogger(__name__)


def run_reconciliation(batch_id: str) -> dict[str, int]:
    """
    Perform full reconciliation for a batch:
    1. Join warehouse.transactions with warehouse.settlements
    2. Detect: MISSING_SETTLEMENT, AMOUNT_MISMATCH, DUPLICATE_SETTLEMENT
    3. Insert issues into reconciliation.reconciliation_issues
    4. Update reconciliation.summary
    5. Return counts per issue_type

    Args:
        batch_id: The batch to reconcile

    Returns:
        Dict of {issue_type: count}
    """
    issue_counts: dict[str, int] = {}

    with managed_connection() as conn:
        # 1. MISSING_SETTLEMENT: transactions with no settlement
        missing_count = _insert_missing_settlement_issues(conn, batch_id)
        issue_counts["MISSING_SETTLEMENT"] = missing_count

        # 2. AMOUNT_MISMATCH: settled amount differs from transaction amount beyond tolerance
        mismatch_count = _insert_amount_mismatch_issues(conn, batch_id)
        issue_counts["AMOUNT_MISMATCH"] = mismatch_count

        # 3. DUPLICATE_SETTLEMENT: same transaction_id has multiple settlements
        dup_stl_count = _insert_duplicate_settlement_issues(conn, batch_id)
        issue_counts["DUPLICATE_SETTLEMENT"] = dup_stl_count

        # 4. ORPHAN_SETTLEMENT: settlements with no matching transaction
        orphan_count = _insert_orphan_settlement_issues(conn, batch_id)
        issue_counts["ORPHAN_SETTLEMENT"] = orphan_count

        # 5. MISSING_FX_RATE: transactions with fx_issue set
        fx_count = _insert_missing_fx_issues(conn, batch_id)
        issue_counts["MISSING_FX_RATE"] = fx_count

        # 6. Update summary
        _upsert_summary(conn, batch_id, issue_counts)

    logger.info("Reconciliation complete for batch=%s: %s", batch_id, issue_counts)
    return issue_counts


def _insert_missing_settlement_issues(conn: psycopg.Connection, batch_id: str) -> int:
    sql = """
    INSERT INTO reconciliation.reconciliation_issues
        (batch_id, transaction_id, settlement_id, issue_type,
         amount_difference, expected_amount, actual_amount, currency, details, detected_at)
    SELECT
        %(batch_id)s,
        t.transaction_id,
        NULL,
        'MISSING_SETTLEMENT',
        NULL,
        t.amount,
        NULL,
        t.currency,
        jsonb_build_object('type', t.type, 'step', t.step),
        NOW()
    FROM warehouse.transactions t
    LEFT JOIN warehouse.settlements s ON t.transaction_id = s.transaction_id
    WHERE t.batch_id = %(batch_id)s
      AND s.settlement_id IS NULL
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"batch_id": batch_id})
        count = cur.rowcount
    logger.info("Inserted %d MISSING_SETTLEMENT issues for batch=%s", count, batch_id)
    return count


def _insert_amount_mismatch_issues(conn: psycopg.Connection, batch_id: str) -> int:
    sql = """
    INSERT INTO reconciliation.reconciliation_issues
        (batch_id, transaction_id, settlement_id, issue_type,
         amount_difference, expected_amount, actual_amount, currency, details, detected_at)
    SELECT
        %(batch_id)s,
        t.transaction_id,
        s.settlement_id,
        'AMOUNT_MISMATCH',
        ABS(t.amount - s.settled_amount),
        t.amount,
        s.settled_amount,
        t.currency,
        jsonb_build_object('difference', ABS(t.amount - s.settled_amount), 'tolerance', %(tol)s),
        NOW()
    FROM warehouse.transactions t
    INNER JOIN warehouse.settlements s ON t.transaction_id = s.transaction_id
    WHERE t.batch_id = %(batch_id)s
      AND ABS(t.amount - s.settled_amount) > %(tol)s::NUMERIC
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"batch_id": batch_id, "tol": str(settings.reconciliation_tolerance)})
        count = cur.rowcount
    logger.info("Inserted %d AMOUNT_MISMATCH issues for batch=%s", count, batch_id)
    return count


def _insert_duplicate_settlement_issues(conn: psycopg.Connection, batch_id: str) -> int:
    sql = """
    INSERT INTO reconciliation.reconciliation_issues
        (batch_id, transaction_id, settlement_id, issue_type,
         amount_difference, expected_amount, actual_amount, currency, details, detected_at)
    SELECT
        %(batch_id)s,
        s.transaction_id,
        s.settlement_id,
        'DUPLICATE_SETTLEMENT',
        NULL,
        NULL,
        s.settled_amount,
        s.settlement_currency,
        jsonb_build_object('settlement_id', s.settlement_id, 'settlement_timestamp', s.settlement_timestamp),
        NOW()
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY transaction_id ORDER BY settlement_timestamp, settlement_id) AS rn
        FROM warehouse.settlements
        WHERE batch_id = %(batch_id)s
    ) s
    WHERE s.rn > 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"batch_id": batch_id})
        count = cur.rowcount
    logger.info("Inserted %d DUPLICATE_SETTLEMENT issues for batch=%s", count, batch_id)
    return count


def _insert_orphan_settlement_issues(conn: psycopg.Connection, batch_id: str) -> int:
    sql = """
    INSERT INTO reconciliation.reconciliation_issues
        (batch_id, transaction_id, settlement_id, issue_type,
         amount_difference, expected_amount, actual_amount, currency, details, detected_at)
    SELECT
        %(batch_id)s,
        s.transaction_id,
        s.settlement_id,
        'ORPHAN_SETTLEMENT',
        NULL,
        NULL,
        s.settled_amount,
        s.settlement_currency,
        jsonb_build_object('settlement_id', s.settlement_id),
        NOW()
    FROM warehouse.settlements s
    LEFT JOIN warehouse.transactions t ON s.transaction_id = t.transaction_id
    WHERE s.batch_id = %(batch_id)s
      AND t.transaction_id IS NULL
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"batch_id": batch_id})
        count = cur.rowcount
    logger.info("Inserted %d ORPHAN_SETTLEMENT issues for batch=%s", count, batch_id)
    return count


def _insert_missing_fx_issues(conn: psycopg.Connection, batch_id: str) -> int:
    """Flag transactions where fx_rate is NULL (FX pair was absent)."""
    sql = """
    INSERT INTO reconciliation.reconciliation_issues
        (batch_id, transaction_id, settlement_id, issue_type,
         amount_difference, expected_amount, actual_amount, currency, details, detected_at)
    SELECT
        %(batch_id)s,
        t.transaction_id,
        NULL,
        'MISSING_FX_RATE',
        NULL,
        t.amount,
        NULL,
        t.currency,
        jsonb_build_object('fx_pair', t.fx_pair, 'currency', t.currency),
        NOW()
    FROM warehouse.transactions t
    WHERE t.batch_id = %(batch_id)s
      AND t.fx_rate IS NULL
      AND t.fx_pair IS NOT NULL
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"batch_id": batch_id})
        count = cur.rowcount
    logger.info("Inserted %d MISSING_FX_RATE issues for batch=%s", count, batch_id)
    return count


def _upsert_summary(
    conn: psycopg.Connection,
    batch_id: str,
    issue_counts: dict[str, int],
) -> None:
    """Upsert reconciliation.summary for this batch."""
    # Compute precision/recall vs anomaly manifest (if available)
    # For now, store raw counts; precision/recall updated later by the precision_recall module
    sql = """
    INSERT INTO reconciliation.summary
        (batch_id, total_transactions, matched, missing_settlement,
         amount_mismatch, duplicate_txn, duplicate_settlement,
         missing_fx_rate, orphan_settlement, delayed_settlement,
         invalid_record, computed_at)
    SELECT
        %(batch_id)s,
        COUNT(*) AS total_transactions,
        COUNT(s.settlement_id) AS matched,
        %(missing_settlement)s,
        %(amount_mismatch)s,
        0,  -- duplicate_txn counted in Spark dedup step
        %(duplicate_settlement)s,
        %(missing_fx_rate)s,
        %(orphan_settlement)s,
        0,  -- delayed_settlement
        0,  -- invalid_record
        NOW()
    FROM warehouse.transactions t
    LEFT JOIN warehouse.settlements s ON t.transaction_id = s.transaction_id
    WHERE t.batch_id = %(batch_id)s
    ON CONFLICT (batch_id) DO UPDATE
        SET
            missing_settlement = EXCLUDED.missing_settlement,
            amount_mismatch = EXCLUDED.amount_mismatch,
            duplicate_settlement = EXCLUDED.duplicate_settlement,
            missing_fx_rate = EXCLUDED.missing_fx_rate,
            orphan_settlement = EXCLUDED.orphan_settlement,
            computed_at = NOW()
    """
    with conn.cursor() as cur:
        cur.execute(sql, {
            "batch_id": batch_id,
            "missing_settlement": issue_counts.get("MISSING_SETTLEMENT", 0),
            "amount_mismatch": issue_counts.get("AMOUNT_MISMATCH", 0),
            "duplicate_settlement": issue_counts.get("DUPLICATE_SETTLEMENT", 0),
            "missing_fx_rate": issue_counts.get("MISSING_FX_RATE", 0),
            "orphan_settlement": issue_counts.get("ORPHAN_SETTLEMENT", 0),
        })
    logger.info("Reconciliation summary upserted for batch=%s", batch_id)


def compute_precision_recall(batch_id: str, manifest_path: str) -> dict[str, float]:
    """
    Compute precision and recall of anomaly detection vs ground-truth manifest.

    Args:
        batch_id: The batch to evaluate
        manifest_path: Path to anomaly_manifest.parquet

    Returns:
        {'precision': float, 'recall': float, 'f1': float}
    """
    import pyarrow.parquet as pq

    # Load manifest
    manifest = pq.read_table(manifest_path).to_pandas()
    true_positives_set = set(manifest["transaction_id"].tolist())

    # Load detected issues
    with managed_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT transaction_id FROM reconciliation.reconciliation_issues WHERE batch_id = %s",
            (batch_id,)
        )
        detected_set = {row[0] for row in cur.fetchall()}

    tp = len(true_positives_set & detected_set)
    fp = len(detected_set - true_positives_set)
    fn = len(true_positives_set - detected_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    result = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
    logger.info("Precision/Recall for batch=%s: %s", batch_id, result)

    # Update summary
    with managed_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
                UPDATE reconciliation.summary
                SET precision_score = %(precision)s, recall_score = %(recall)s
                WHERE batch_id = %(batch_id)s
                """,
            {**result, "batch_id": batch_id}
        )

    return result

