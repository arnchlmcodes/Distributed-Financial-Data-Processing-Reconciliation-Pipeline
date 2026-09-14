"""
tests/unit/test_database.py — Unit tests for M4 database layer (no live DB required).
"""
from __future__ import annotations

from unittest import mock


def test_ddl_files_exist():
    """All four DDL files must be present on disk."""
    from src.database.schema_manager import DDL_FILES

    for f in DDL_FILES:
        assert f.exists(), f"Missing DDL file: {f}"


def test_schema_manager_importable():
    """schema_manager module must expose apply_ddl."""
    from src.database import schema_manager

    assert hasattr(schema_manager, "apply_ddl")


def test_get_connection_calls_psycopg_connect():
    """get_connection() must call psycopg.connect with the DSN from settings."""
    with mock.patch("src.database.connection.psycopg.connect") as mock_connect:
        mock_connect.return_value = mock.MagicMock()
        from src.database.connection import get_connection

        conn = get_connection()
        mock_connect.assert_called_once()
        assert conn is not None


def test_managed_connection_commits_on_success():
    """managed_connection() must commit when no exception is raised."""
    fake_conn = mock.MagicMock()
    with mock.patch("src.database.connection.get_connection", return_value=fake_conn):
        from src.database.connection import managed_connection

        with managed_connection():
            pass

        fake_conn.commit.assert_called_once()
        fake_conn.close.assert_called_once()


def test_managed_connection_rolls_back_on_error():
    """managed_connection() must rollback when an exception is raised."""
    fake_conn = mock.MagicMock()
    with mock.patch("src.database.connection.get_connection", return_value=fake_conn):
        from src.database.connection import managed_connection

        try:
            with managed_connection():
                raise ValueError("boom")
        except ValueError:
            pass

        fake_conn.rollback.assert_called_once()
        fake_conn.commit.assert_not_called()
        fake_conn.close.assert_called_once()
