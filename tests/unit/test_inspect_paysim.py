"""
test_inspect_paysim.py — Unit tests for src/generators/inspect_paysim.py

Creates a tiny 10-row CSV fixture in a temp directory and validates that
profile_dataset() returns the expected structure and values.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Make sure the repo root is on sys.path so imports work regardless of how
# pytest is invoked.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.generators.inspect_paysim import profile_dataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

_SAMPLE_ROWS = [
    # step, type, amount,     nameOrig, oldOrg,  newOrig, nameDest, oldDest, newDest, isFraud, isFlagged
    ("1", "PAYMENT",  "100.00", "C001", "500.00", "400.00", "M001", "0.00", "100.00", "0", "0"),
    ("1", "PAYMENT",  "200.50", "C002", "600.00", "399.50", "M002", "0.00", "200.50", "0", "0"),
    ("2", "TRANSFER", "300.75", "C003", "300.75", "0.00",   "C004", "0.00", "300.75", "1", "0"),
    ("2", "CASH_OUT", "50.00",  "C005", "200.00", "150.00", "C006", "0.00", "50.00",  "0", "0"),
    ("3", "TRANSFER", "999.99", "C007", "999.99", "0.00",   "C008", "0.00", "999.99", "1", "1"),
    ("3", "PAYMENT",  "10.00",  "C009", "100.00", "90.00",  "M003", "0.00", "10.00",  "0", "0"),
    ("4", "CASH_IN",  "750.00", "C010", "0.00",   "750.00", "C011", "750.00","0.00", "0", "0"),
    ("4", "DEBIT",    "25.25",  "C012", "500.00", "474.75", "C013", "0.00", "25.25",  "0", "0"),
    ("5", "CASH_OUT", "88.88",  "C014", "88.88",  "0.00",   "C015", "0.00", "88.88",  "0", "0"),
    # Last row: duplicate of row index 0 (same step+type+amount+nameOrig+nameDest)
    ("1", "PAYMENT",  "100.00", "C001", "500.00", "400.00", "M001", "0.00", "100.00", "0", "0"),
]


@pytest.fixture()
def tiny_csv(tmp_path: Path) -> Path:
    """Write a 10-row PaySim-style CSV to *tmp_path* and return the path."""
    csv_file = tmp_path / "paysim_tiny.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_COLUMNS)
        writer.writerows(_SAMPLE_ROWS)
    return csv_file


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProfileDataset:
    """Tests for profile_dataset()."""

    def test_returns_dict(self, tiny_csv: Path) -> None:
        """profile_dataset must return a dict."""
        result = profile_dataset(tiny_csv, chunk_size=5)
        assert isinstance(result, dict)

    def test_required_keys_present(self, tiny_csv: Path) -> None:
        """All required top-level keys must be present."""
        result = profile_dataset(tiny_csv, chunk_size=5)
        required = {
            "row_count",
            "column_types",
            "null_counts",
            "type_distribution",
            "fraud_distribution",
            "amount_stats",
            "duplicate_count",
        }
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - set(result.keys())}"
        )

    def test_row_count(self, tiny_csv: Path) -> None:
        """row_count must equal the number of data rows (10)."""
        result = profile_dataset(tiny_csv, chunk_size=5)
        assert result["row_count"] == 10

    def test_amount_stats_are_strings(self, tiny_csv: Path) -> None:
        """amount_stats min/max/mean must be strings, not floats."""
        result = profile_dataset(tiny_csv, chunk_size=5)
        stats = result["amount_stats"]
        assert set(stats.keys()) == {"min", "max", "mean"}, (
            f"Unexpected amount_stats keys: {stats.keys()}"
        )
        for key in ("min", "max", "mean"):
            assert isinstance(stats[key], str), (
                f"amount_stats['{key}'] is {type(stats[key])}, expected str"
            )
            # Must not be a Python float repr — no 'e' notation, no NaN, no inf
            assert stats[key] not in ("nan", "inf", "-inf"), (
                f"amount_stats['{key}'] has invalid value: {stats[key]}"
            )

    def test_type_distribution_keys(self, tiny_csv: Path) -> None:
        """type_distribution must contain the transaction types in the fixture."""
        result = profile_dataset(tiny_csv, chunk_size=5)
        dist = result["type_distribution"]
        # Our fixture has PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT
        for tx_type in ("PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"):
            assert tx_type in dist, f"'{tx_type}' missing from type_distribution"

    def test_fraud_distribution(self, tiny_csv: Path) -> None:
        """fraud_distribution must include both 0 and 1 entries."""
        result = profile_dataset(tiny_csv, chunk_size=5)
        fraud = result["fraud_distribution"]
        # Keys are stored as strings (JSON-safe)
        assert "0" in fraud or 0 in fraud, "No non-fraud rows in fraud_distribution"
        assert "1" in fraud or 1 in fraud, "No fraud rows in fraud_distribution"

    def test_duplicate_count(self, tiny_csv: Path) -> None:
        """The 10th row is a duplicate of the 1st — duplicate_count must be >= 1."""
        result = profile_dataset(tiny_csv, chunk_size=5)
        assert result["duplicate_count"] >= 1, (
            f"Expected at least 1 duplicate, got {result['duplicate_count']}"
        )

    def test_column_types_is_dict(self, tiny_csv: Path) -> None:
        """column_types must be a dict with a key per CSV column."""
        result = profile_dataset(tiny_csv, chunk_size=5)
        col_types = result["column_types"]
        assert isinstance(col_types, dict)
        for col in _CSV_COLUMNS:
            assert col in col_types, f"'{col}' missing from column_types"

    def test_null_counts_dict(self, tiny_csv: Path) -> None:
        """null_counts must be a dict (may be empty if no nulls in fixture)."""
        result = profile_dataset(tiny_csv, chunk_size=5)
        assert isinstance(result["null_counts"], dict)

    def test_amount_min_less_than_max(self, tiny_csv: Path) -> None:
        """amount_stats min must be <= max (as Decimal comparison)."""
        from decimal import Decimal

        result = profile_dataset(tiny_csv, chunk_size=5)
        stats = result["amount_stats"]
        assert Decimal(stats["min"]) <= Decimal(stats["max"])

    def test_chunk_boundary_consistency(self, tiny_csv: Path) -> None:
        """Row count must be the same regardless of chunk size."""
        r1 = profile_dataset(tiny_csv, chunk_size=3)
        r2 = profile_dataset(tiny_csv, chunk_size=10)
        assert r1["row_count"] == r2["row_count"] == 10
