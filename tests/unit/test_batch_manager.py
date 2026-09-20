"""
test_batch_manager.py — Unit tests for batch tracking and idempotency.
"""
from __future__ import annotations

from unittest import mock

import pytest

from src.database.batch_manager import (
    BatchAlreadyCompletedError,
    generate_batch_id,
    register_batch,
)


def test_generate_batch_id():
    batch_id = generate_batch_id("data/raw/paysim.csv")
    assert batch_id.startswith("paysim_")
    assert len(batch_id) > 10


def test_register_batch_idempotency():
    mock_cursor = mock.MagicMock()
    # Simulate batch is already COMPLETED
    mock_cursor.fetchone.return_value = ("COMPLETED",)
    mock_conn = mock.MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    with pytest.raises(BatchAlreadyCompletedError):
        register_batch("batch_001", "source.csv", conn=mock_conn)


def test_register_batch_new():
    mock_cursor = mock.MagicMock()
    # Batch does not exist yet
    mock_cursor.fetchone.return_value = None
    mock_conn = mock.MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    register_batch("batch_002", "source.csv", conn=mock_conn)
    assert mock_cursor.execute.call_count == 2  # SELECT then INSERT

