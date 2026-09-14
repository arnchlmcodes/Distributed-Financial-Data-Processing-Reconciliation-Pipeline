"""
schema_manager.py — Applies all DDL scripts idempotently.
Run: python -m src.database.schema_manager
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.database.connection import managed_connection

logger = logging.getLogger(__name__)

SQL_DIR = Path(__file__).resolve().parent.parent.parent / "sql"

DDL_FILES = [
    SQL_DIR / "staging" / "create_staging.sql",
    SQL_DIR / "warehouse" / "create_warehouse.sql",
    SQL_DIR / "reconciliation" / "create_reconciliation.sql",
    SQL_DIR / "metadata" / "create_metadata.sql",
]


def apply_ddl() -> None:
    """Execute all DDL files in order. Safe to run multiple times (IF NOT EXISTS)."""
    with managed_connection() as conn:
        for ddl_file in DDL_FILES:
            logger.info("Applying DDL: %s", ddl_file.name)
            sql = ddl_file.read_text(encoding="utf-8")
            with conn.cursor() as cur:
                cur.execute(sql)
        logger.info("All DDL applied successfully.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    apply_ddl()
    print("Schema applied.")
