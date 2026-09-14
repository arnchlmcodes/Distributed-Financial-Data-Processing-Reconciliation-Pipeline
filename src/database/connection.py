"""
connection.py — psycopg3 connection helper.
Loads credentials from .env via src.config.settings.
"""
from __future__ import annotations

import contextlib
import logging
from collections.abc import Generator

import psycopg

from src.config import settings

logger = logging.getLogger(__name__)


def get_connection() -> psycopg.Connection:
    """Return an autocommit=False psycopg3 connection."""
    conn = psycopg.connect(settings.pg_dsn)
    logger.debug(
        "DB connection opened: %s@%s:%s/%s",
        settings.pg_user,
        settings.pg_host,
        settings.pg_port,
        settings.pg_db,
    )
    return conn


@contextlib.contextmanager
def managed_connection() -> Generator[psycopg.Connection, None, None]:
    """Context manager: commits on success, rolls back on exception."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
