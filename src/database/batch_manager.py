"""
batch_manager.py — Manage pipeline batch state in metadata.pipeline_batches.

Provides idempotency: before processing, check if batch already COMPLETED.
Records batch lifecycle: PENDING → RUNNING → COMPLETED / FAILED.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime

import psycopg

from src.database.connection import managed_connection

logger = logging.getLogger(__name__)


class BatchAlreadyCompletedError(Exception):
    """Raised when attempting to re-process an already-completed batch."""


def generate_batch_id(source_file: str) -> str:
    """Generate a deterministic batch ID from source file + timestamp."""
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
    safe_name = Path(source_file).stem if source_file else "unknown"
    return f"{safe_name}_{ts}"


def register_batch(
    batch_id: str,
    source_file: str,
    conn: psycopg.Connection | None = None,
) -> None:
    """
    Register a new batch in metadata.pipeline_batches with status=PENDING.
    If batch_id already exists and is COMPLETED, raises BatchAlreadyCompletedError.
    """
    check_sql = "SELECT status FROM metadata.pipeline_batches WHERE batch_id = %s"
    insert_sql = """
    INSERT INTO metadata.pipeline_batches (batch_id, source_file, status, started_at)
    VALUES (%s, %s, 'PENDING', NOW())
    ON CONFLICT (batch_id) DO NOTHING
    """

    def _run(c: psycopg.Connection) -> None:
        with c.cursor() as cur:
            cur.execute(check_sql, (batch_id,))
            row = cur.fetchone()
            if row and row[0] == "COMPLETED":
                raise BatchAlreadyCompletedError(
                    f"Batch {batch_id!r} already COMPLETED. Re-run skipped (idempotent)."
                )
            cur.execute(insert_sql, (batch_id, source_file))
        logger.info("Batch registered: batch_id=%s, source=%s", batch_id, source_file)

    if conn is not None:
        _run(conn)
    else:
        with managed_connection() as c:
            _run(c)


def update_batch_status(
    batch_id: str,
    status: str,
    rows_ingested: int | None = None,
    rows_valid: int | None = None,
    rows_rejected: int | None = None,
    rows_loaded_warehouse: int | None = None,
    error_message: str | None = None,
    conn: psycopg.Connection | None = None,
) -> None:
    """
    Update batch status and metrics.
    status must be one of: PENDING, RUNNING, COMPLETED, FAILED.
    """
    update_sql = """
    UPDATE metadata.pipeline_batches
    SET
        status = %(status)s,
        rows_ingested = COALESCE(%(rows_ingested)s, rows_ingested),
        rows_valid = COALESCE(%(rows_valid)s, rows_valid),
        rows_rejected = COALESCE(%(rows_rejected)s, rows_rejected),
        rows_loaded_warehouse = COALESCE(%(rows_loaded_warehouse)s, rows_loaded_warehouse),
        error_message = COALESCE(%(error_message)s, error_message),
        completed_at = CASE WHEN %(status)s IN ('COMPLETED', 'FAILED') THEN NOW() ELSE completed_at END
    WHERE batch_id = %(batch_id)s
    """
    params = {
        "batch_id": batch_id,
        "status": status,
        "rows_ingested": rows_ingested,
        "rows_valid": rows_valid,
        "rows_rejected": rows_rejected,
        "rows_loaded_warehouse": rows_loaded_warehouse,
        "error_message": error_message,
    }

    def _run(c: psycopg.Connection) -> None:
        with c.cursor() as cur:
            cur.execute(update_sql, params)
        logger.info("Batch %s updated: status=%s", batch_id, status)

    if conn is not None:
        _run(conn)
    else:
        with managed_connection() as c:
            _run(c)


def get_batch_status(batch_id: str) -> dict | None:
    """Retrieve batch metadata for a given batch_id. Returns None if not found."""
    query_sql = """
    SELECT batch_id, source_file, status, rows_ingested, rows_valid, rows_rejected,
           rows_loaded_warehouse, started_at, completed_at, error_message, retry_count
    FROM metadata.pipeline_batches
    WHERE batch_id = %s
    """
    with managed_connection() as conn, conn.cursor() as cur:
        cur.execute(query_sql, (batch_id,))
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))


def list_completed_batches() -> list[str]:
    """Return all batch_ids with status=COMPLETED."""
    query_sql = "SELECT batch_id FROM metadata.pipeline_batches WHERE status = 'COMPLETED' ORDER BY started_at"
    with managed_connection() as conn, conn.cursor() as cur:
        cur.execute(query_sql)
        return [row[0] for row in cur.fetchall()]


# Make Path available (used in generate_batch_id)
from pathlib import Path

