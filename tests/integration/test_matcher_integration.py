"""
test_matcher_integration.py — Integration tests for reconciliation matcher.
"""
from __future__ import annotations

from unittest import mock

from src.reconciliation.matcher import (
    _insert_amount_mismatch_issues,
    _insert_missing_settlement_issues,
    run_reconciliation,
)


def test_insert_missing_settlement_issues():
    mock_cursor = mock.MagicMock()
    mock_cursor.rowcount = 5
    mock_conn = mock.MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    count = _insert_missing_settlement_issues(mock_conn, batch_id="b1")
    assert count == 5
    assert mock_cursor.execute.called


def test_insert_amount_mismatch_issues():
    mock_cursor = mock.MagicMock()
    mock_cursor.rowcount = 3
    mock_conn = mock.MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    count = _insert_amount_mismatch_issues(mock_conn, batch_id="b1")
    assert count == 3
    assert mock_cursor.execute.called


def test_run_reconciliation():
    with mock.patch("src.reconciliation.matcher.managed_connection") as mock_mgr:
        mock_conn = mock.MagicMock()
        mock_cursor = mock.MagicMock()
        mock_cursor.rowcount = 2
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_mgr.return_value.__enter__.return_value = mock_conn

        issues = run_reconciliation(batch_id="b1")
        assert "MISSING_SETTLEMENT" in issues
        assert "AMOUNT_MISMATCH" in issues
        assert "DUPLICATE_SETTLEMENT" in issues

