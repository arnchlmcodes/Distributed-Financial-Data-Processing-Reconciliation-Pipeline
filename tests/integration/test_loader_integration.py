"""
test_loader_integration.py — Integration tests for bulk loader.
Tests COPY streaming to staging and upsert to warehouse.
"""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

from src.database.loader import (
    copy_transactions_to_staging,
    upsert_transactions_to_warehouse,
)


def test_copy_transactions_to_staging(tmp_path: Path):
    # Create sample transactions parquet
    schema = pa.schema([
        ("batch_id", pa.string()),
        ("step", pa.int32()),
        ("type", pa.string()),
        ("amount", pa.string()),
        ("name_orig", pa.string()),
        ("old_balance_orig", pa.string()),
        ("new_balance_orig", pa.string()),
        ("name_dest", pa.string()),
        ("old_balance_dest", pa.string()),
        ("new_balance_dest", pa.string()),
        ("is_fraud", pa.int32()),
        ("is_flagged_fraud", pa.int32()),
        ("transaction_id", pa.string()),
        ("currency", pa.string()),
        ("fx_pair", pa.string()),
        ("transaction_timestamp", pa.string()),
    ])
    data = {
        "batch_id": ["b1", "b1"],
        "step": [1, 2],
        "type": ["PAYMENT", "TRANSFER"],
        "amount": ["100.00", "500.00"],
        "name_orig": ["C1", "C2"],
        "old_balance_orig": ["1000.00", "500.00"],
        "new_balance_orig": ["900.00", "0.00"],
        "name_dest": ["M1", "C3"],
        "old_balance_dest": ["0.00", "100.00"],
        "new_balance_dest": ["0.00", "600.00"],
        "is_fraud": [0, 0],
        "is_flagged_fraud": [0, 0],
        "transaction_id": ["tx-1", "tx-2"],
        "currency": ["USD", "EUR"],
        "fx_pair": ["USD/USD", "EUR/USD"],
        "transaction_timestamp": ["2023-01-01T01:00:00", "2023-01-01T02:00:00"],
    }
    table = pa.Table.from_pydict(data, schema=schema)
    parquet_file = tmp_path / "txns.parquet"
    pq.write_table(table, parquet_file)

    mock_copy_obj = mock.MagicMock()
    mock_cursor = mock.MagicMock()
    mock_cursor.copy.return_value.__enter__.return_value = mock_copy_obj

    mock_conn = mock.MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    copied = copy_transactions_to_staging(mock_conn, parquet_file, batch_id="b1")
    assert copied == 2
    assert mock_copy_obj.write.called


def test_upsert_transactions_to_warehouse():
    mock_cursor = mock.MagicMock()
    mock_cursor.rowcount = 10
    mock_conn = mock.MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    count = upsert_transactions_to_warehouse(mock_conn, batch_id="b1")
    assert count == 10
    assert mock_cursor.execute.called

