"""
queries.py — Analytical SQL queries for warehouse and reconciliation metrics.

Contains queries for:
- Daily transaction volume by type
- Currency distribution and USD converted totals
- Top origin and destination accounts
- Reconciliation issue breakdown and trends
- Anomaly rates by batch
"""
from __future__ import annotations

import logging
from typing import Any

from src.database.connection import managed_connection

logger = logging.getLogger(__name__)


DAILY_VOLUME_BY_TYPE_SQL = """
SELECT
    DATE_TRUNC('day', transaction_timestamp) AS txn_date,
    type,
    COUNT(*) AS total_transactions,
    SUM(amount) AS total_amount,
    AVG(amount) AS avg_amount,
    MIN(amount) AS min_amount,
    MAX(amount) AS max_amount
FROM warehouse.transactions
WHERE batch_id = %(batch_id)s
GROUP BY 1, 2
ORDER BY 1, 2;
"""

CURRENCY_SUMMARY_SQL = """
SELECT
    currency,
    COUNT(*) AS transaction_count,
    SUM(amount) AS total_original_amount,
    SUM(amount_usd) AS total_usd_amount,
    AVG(fx_rate) AS avg_fx_rate
FROM warehouse.transactions
WHERE batch_id = %(batch_id)s
GROUP BY currency
ORDER BY total_usd_amount DESC NULLS LAST;
"""

TOP_ACCOUNTS_SQL = """
SELECT
    name_orig,
    COUNT(*) AS outbound_transactions,
    SUM(amount) AS total_outbound_amount
FROM warehouse.transactions
WHERE batch_id = %(batch_id)s
GROUP BY name_orig
ORDER BY total_outbound_amount DESC
LIMIT %(limit)s;
"""

ISSUE_BREAKDOWN_SQL = """
SELECT
    issue_type,
    resolution_status,
    COUNT(*) AS count,
    SUM(COALESCE(amount_difference, 0)) AS total_amount_difference
FROM reconciliation.reconciliation_issues
WHERE batch_id = %(batch_id)s
GROUP BY issue_type, resolution_status
ORDER BY count DESC;
"""

BATCH_ANOMALY_SUMMARY_SQL = """
SELECT
    s.batch_id,
    s.total_transactions,
    s.matched,
    s.missing_settlement,
    s.amount_mismatch,
    s.duplicate_settlement,
    s.missing_fx_rate,
    s.orphan_settlement,
    s.precision_score,
    s.recall_score
FROM reconciliation.summary s
WHERE s.batch_id = %(batch_id)s;
"""


def execute_query(sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Execute a read-only analytical query and return list of dicts."""
    with managed_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params or {})
        cols = [desc[0] for desc in cur.description] if cur.description else []
        rows = cur.fetchall()
        return [dict(zip(cols, row)) for row in rows]
