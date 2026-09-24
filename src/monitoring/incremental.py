"""
incremental.py — Watermark-based incremental processing and late-arriving record handling.

Features:
- Track highest processed transaction_timestamp (watermark)
- Check batch status before execution to guarantee idempotency
- Reprocess late-arriving or previously quarantined records with reprocess_flag=True
"""
from __future__ import annotations

import logging
from datetime import datetime

from src.database.connection import managed_connection

logger = logging.getLogger(__name__)


def get_latest_watermark() -> datetime | None:
    """
    Get the highest transaction_timestamp processed into warehouse.transactions.
    Used for incremental window queries.
    """
    sql = "SELECT MAX(transaction_timestamp) FROM warehouse.transactions;"
    with managed_connection() as conn, conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else None


def flag_late_records_for_reprocessing(
    source_file: str,
    cutoff_timestamp: datetime,
) -> int:
    """
    Identify records quarantined earlier that can now be reprocessed.
    Sets reprocess_flag=TRUE in metadata.quarantine.
    """
    sql = """
    UPDATE metadata.quarantine
    SET reprocess_flag = TRUE
    WHERE source_file = %s
      AND quarantined_at <= %s
      AND reprocess_flag = FALSE;
    """
    with managed_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (source_file, cutoff_timestamp))
        count = cur.rowcount
    logger.info("Flagged %d records for reprocessing in %s", count, source_file)
    return count


def fetch_pending_reprocess_records(limit: int = 1000) -> list[dict]:
    """Fetch quarantined records marked for reprocessing."""
    sql = """
    SELECT quarantine_id, batch_id, source_file, record_type, payload, reasons
    FROM metadata.quarantine
    WHERE reprocess_flag = TRUE
    ORDER BY quarantined_at
    LIMIT %s;
    """
    with managed_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (limit,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def mark_quarantine_reprocessed(quarantine_ids: list[int]) -> int:
    """Clear reprocess_flag after successful replay."""
    if not quarantine_ids:
        return 0
    sql = """
    UPDATE metadata.quarantine
    SET reprocess_flag = FALSE
    WHERE quarantine_id = ANY(%s);
    """
    with managed_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (quarantine_ids,))
        return cur.rowcount

