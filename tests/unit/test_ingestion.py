"""
test_ingestion.py — Unit tests for M3: ingestion schema, reader, and validator.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.ingestion.reader import read_paysim_chunks
from src.ingestion.validator import (
    VALID_TYPES,
    ValidationResult,
    validate_chunk,
    write_quarantine,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_row(**overrides) -> dict:
    """Return a minimal valid raw PaySim row dict."""
    base = {
        "transaction_id": "T_001",
        "batch_id": "batch_test",
        "step": "1",
        "type": "PAYMENT",
        "amount": "250.00",
        "nameOrig": "C123456789",
        "oldbalanceOrg": "1000.00",
        "newbalanceOrig": "750.00",
        "nameDest": "M987654321",
        "oldbalanceDest": "0.00",
        "newbalanceDest": "250.00",
        "isFraud": "0",
        "isFlaggedFraud": "0",
    }
    base.update(overrides)
    return base


def _make_csv(tmp_path: Path, rows: list[dict], filename: str = "test.csv") -> Path:
    """Write a list of dicts to a CSV file and return its Path."""
    if not rows:
        raise ValueError("rows must be non-empty to infer fieldnames")
    out = tmp_path / filename
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out


# ---------------------------------------------------------------------------
# validate_chunk — happy path
# ---------------------------------------------------------------------------

class TestValidateChunkValid:
    def test_all_valid_rows_go_to_valid_list(self):
        rows = [_make_valid_row(transaction_id=f"T_{i:03d}") for i in range(5)]
        result = validate_chunk(rows, batch_id="b1", source_file="test.csv")

        assert len(result.valid_rows) == 5
        assert len(result.rejected_rows) == 0
        assert result.rejection_counts == {}

    def test_valid_result_type(self):
        result = validate_chunk([_make_valid_row()], batch_id="b1", source_file="f.csv")
        assert isinstance(result, ValidationResult)

    @pytest.mark.parametrize("txn_type", list(VALID_TYPES))
    def test_all_valid_types_accepted(self, txn_type: str):
        row = _make_valid_row(type=txn_type)
        result = validate_chunk([row], batch_id="b1", source_file="f.csv")
        assert len(result.valid_rows) == 1
        assert len(result.rejected_rows) == 0


# ---------------------------------------------------------------------------
# validate_chunk — missing transaction_id
# ---------------------------------------------------------------------------

class TestValidateChunkMissingTransactionId:
    def test_none_transaction_id_is_rejected(self):
        row = _make_valid_row(transaction_id=None)
        result = validate_chunk([row], batch_id="b1", source_file="f.csv")

        assert len(result.valid_rows) == 0
        assert len(result.rejected_rows) == 1

    def test_rejection_reason_contains_missing_transaction_id(self):
        row = _make_valid_row(transaction_id=None)
        result = validate_chunk([row], batch_id="b1", source_file="f.csv")

        reasons = json.loads(result.rejected_rows[0]["reasons"])
        assert any("missing transaction_id" in r for r in reasons)

    def test_empty_string_transaction_id_is_rejected(self):
        row = _make_valid_row(transaction_id="   ")
        result = validate_chunk([row], batch_id="b1", source_file="f.csv")
        assert len(result.rejected_rows) == 1

    def test_rejection_count_incremented(self):
        rows = [_make_valid_row(transaction_id=None) for _ in range(3)]
        result = validate_chunk(rows, batch_id="b1", source_file="f.csv")
        assert result.rejection_counts.get("missing transaction_id") == 3


# ---------------------------------------------------------------------------
# validate_chunk — negative amount
# ---------------------------------------------------------------------------

class TestValidateChunkNegativeAmount:
    def test_negative_amount_is_rejected(self):
        row = _make_valid_row(amount="-100.00")
        result = validate_chunk([row], batch_id="b1", source_file="f.csv")

        assert len(result.valid_rows) == 0
        assert len(result.rejected_rows) == 1

    def test_rejection_reason_contains_negative_amount(self):
        row = _make_valid_row(amount="-0.01")
        result = validate_chunk([row], batch_id="b1", source_file="f.csv")

        reasons = json.loads(result.rejected_rows[0]["reasons"])
        assert any("negative amount" in r for r in reasons)

    def test_zero_amount_is_valid(self):
        """Zero is a valid amount (not negative)."""
        row = _make_valid_row(amount="0.00")
        result = validate_chunk([row], batch_id="b1", source_file="f.csv")
        assert len(result.valid_rows) == 1

    def test_missing_amount_is_rejected(self):
        row = _make_valid_row(amount=None)
        result = validate_chunk([row], batch_id="b1", source_file="f.csv")
        reasons = json.loads(result.rejected_rows[0]["reasons"])
        assert any("missing amount" in r for r in reasons)

    def test_unparseable_amount_is_rejected(self):
        row = _make_valid_row(amount="not_a_number")
        result = validate_chunk([row], batch_id="b1", source_file="f.csv")
        reasons = json.loads(result.rejected_rows[0]["reasons"])
        assert any("unparseable amount" in r for r in reasons)


# ---------------------------------------------------------------------------
# validate_chunk — invalid type
# ---------------------------------------------------------------------------

class TestValidateChunkInvalidType:
    def test_unknown_type_is_rejected(self):
        row = _make_valid_row(type="WIRE_TRANSFER")
        result = validate_chunk([row], batch_id="b1", source_file="f.csv")
        assert len(result.rejected_rows) == 1

    def test_rejection_reason_contains_invalid_type(self):
        row = _make_valid_row(type="UNKNOWN")
        result = validate_chunk([row], batch_id="b1", source_file="f.csv")
        reasons = json.loads(result.rejected_rows[0]["reasons"])
        assert any("invalid type" in r for r in reasons)

    def test_empty_type_is_rejected(self):
        row = _make_valid_row(type="")
        result = validate_chunk([row], batch_id="b1", source_file="f.csv")
        assert len(result.rejected_rows) == 1

    def test_case_sensitive_type(self):
        """Type matching is case-sensitive; lowercase should fail."""
        row = _make_valid_row(type="payment")
        result = validate_chunk([row], batch_id="b1", source_file="f.csv")
        assert len(result.rejected_rows) == 1


# ---------------------------------------------------------------------------
# validate_chunk — quarantine payload integrity
# ---------------------------------------------------------------------------

class TestQuarantinePayloadIntegrity:
    def test_quarantine_record_has_all_required_keys(self):
        row = _make_valid_row(transaction_id=None)
        result = validate_chunk([row], batch_id="b_integrity", source_file="src.csv")

        rec = result.rejected_rows[0]
        for key in ("batch_id", "source_file", "record_type", "payload", "reasons", "quarantined_at"):
            assert key in rec, f"Missing quarantine key: {key}"

    def test_quarantine_payload_is_valid_json(self):
        row = _make_valid_row(amount="-1")
        result = validate_chunk([row], batch_id="b1", source_file="f.csv")
        payload = json.loads(result.rejected_rows[0]["payload"])
        assert isinstance(payload, dict)

    def test_quarantine_reasons_is_json_list(self):
        row = _make_valid_row(transaction_id=None, amount="-5")
        result = validate_chunk([row], batch_id="b1", source_file="f.csv")
        reasons = json.loads(result.rejected_rows[0]["reasons"])
        assert isinstance(reasons, list)
        assert len(reasons) >= 2  # at least two failures

    def test_batch_id_propagated(self):
        row = _make_valid_row(transaction_id=None)
        result = validate_chunk([row], batch_id="my_batch_99", source_file="f.csv")
        assert result.rejected_rows[0]["batch_id"] == "my_batch_99"

    def test_source_file_propagated(self):
        row = _make_valid_row(transaction_id=None)
        result = validate_chunk([row], batch_id="b1", source_file="special_file.csv")
        assert result.rejected_rows[0]["source_file"] == "special_file.csv"

    def test_record_type_is_transaction(self):
        row = _make_valid_row(transaction_id=None)
        result = validate_chunk([row], batch_id="b1", source_file="f.csv")
        assert result.rejected_rows[0]["record_type"] == "TRANSACTION"


# ---------------------------------------------------------------------------
# write_quarantine
# ---------------------------------------------------------------------------

class TestWriteQuarantine:
    def test_returns_none_for_empty_list(self, tmp_path: Path):
        result = write_quarantine([], batch_id="b1", output_dir=tmp_path)
        assert result is None

    def test_returns_path_for_non_empty(self, tmp_path: Path):
        row = _make_valid_row(transaction_id=None)
        vr = validate_chunk([row], batch_id="b_write", source_file="f.csv")
        out = write_quarantine(vr.rejected_rows, batch_id="b_write", output_dir=tmp_path)
        assert out is not None
        assert out.exists()

    def test_parquet_has_correct_schema_columns(self, tmp_path: Path):
        row = _make_valid_row(transaction_id=None)
        vr = validate_chunk([row], batch_id="b_schema", source_file="f.csv")
        out = write_quarantine(vr.rejected_rows, batch_id="b_schema", output_dir=tmp_path)

        table = pq.read_table(out)
        expected_cols = {"batch_id", "source_file", "record_type", "payload", "reasons", "quarantined_at"}
        assert expected_cols.issubset(set(table.schema.names))

    def test_parquet_row_count_matches(self, tmp_path: Path):
        rows = [_make_valid_row(transaction_id=None) for _ in range(7)]
        vr = validate_chunk(rows, batch_id="b_count", source_file="f.csv")
        out = write_quarantine(vr.rejected_rows, batch_id="b_count", output_dir=tmp_path)

        table = pq.read_table(out)
        assert table.num_rows == 7

    def test_parquet_reasons_is_valid_json_list(self, tmp_path: Path):
        row = _make_valid_row(amount="-99")
        vr = validate_chunk([row], batch_id="b_r", source_file="f.csv")
        out = write_quarantine(vr.rejected_rows, batch_id="b_r", output_dir=tmp_path)

        table = pq.read_table(out)
        reasons_str = table.column("reasons")[0].as_py()
        reasons = json.loads(reasons_str)
        assert isinstance(reasons, list)
        assert len(reasons) > 0

    def test_output_dir_created_if_missing(self, tmp_path: Path):
        new_dir = tmp_path / "nested" / "quarantine"
        assert not new_dir.exists()
        row = _make_valid_row(transaction_id=None)
        vr = validate_chunk([row], batch_id="b_mkdir", source_file="f.csv")
        write_quarantine(vr.rejected_rows, batch_id="b_mkdir", output_dir=new_dir)
        assert new_dir.exists()

    def test_filename_contains_batch_id(self, tmp_path: Path):
        row = _make_valid_row(transaction_id=None)
        vr = validate_chunk([row], batch_id="batch_xyz", source_file="f.csv")
        out = write_quarantine(vr.rejected_rows, batch_id="batch_xyz", output_dir=tmp_path)
        assert "batch_xyz" in out.name


# ---------------------------------------------------------------------------
# read_paysim_chunks
# ---------------------------------------------------------------------------

# PaySim CSV columns (must be present in the test CSV header)
_PAYSIM_CSV_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
    "isFraud", "isFlaggedFraud",
]


def _make_paysim_csv_row(i: int) -> dict:
    return {
        "step": str(i),
        "type": "PAYMENT",
        "amount": f"{100 + i}.00",
        "nameOrig": f"C{i:09d}",
        "oldbalanceOrg": "5000.00",
        "newbalanceOrig": f"{5000 - 100 - i:.2f}",
        "nameDest": f"M{i:09d}",
        "oldbalanceDest": "0.00",
        "newbalanceDest": f"{100 + i:.2f}",
        "isFraud": "0",
        "isFlaggedFraud": "0",
    }


class TestReadPaySimChunks:
    def test_yields_correct_total_rows_for_5_rows(self, tmp_path: Path):
        raw_rows = [_make_paysim_csv_row(i) for i in range(5)]
        csv_path = _make_csv(tmp_path, raw_rows, filename="paysim_test.csv")

        all_rows = [row for chunk in read_paysim_chunks(csv_path, chunk_size=50_000) for row in chunk]
        assert len(all_rows) == 5

    def test_chunk_size_controls_chunk_length(self, tmp_path: Path):
        raw_rows = [_make_paysim_csv_row(i) for i in range(5)]
        csv_path = _make_csv(tmp_path, raw_rows)

        chunks = list(read_paysim_chunks(csv_path, chunk_size=2))
        # 5 rows with chunk_size=2 → chunks of size [2, 2, 1]
        assert len(chunks) == 3
        assert [len(c) for c in chunks] == [2, 2, 1]

    def test_row_dicts_have_expected_keys(self, tmp_path: Path):
        raw_rows = [_make_paysim_csv_row(0)]
        csv_path = _make_csv(tmp_path, raw_rows)

        chunks = list(read_paysim_chunks(csv_path, batch_id="b_keys"))
        row = chunks[0][0]
        for col in _PAYSIM_CSV_COLUMNS:
            assert col in row, f"Missing column in row dict: {col}"

    def test_batch_id_attached_to_each_row(self, tmp_path: Path):
        raw_rows = [_make_paysim_csv_row(i) for i in range(3)]
        csv_path = _make_csv(tmp_path, raw_rows)

        chunks = list(read_paysim_chunks(csv_path, batch_id="my_batch"))
        for chunk in chunks:
            for row in chunk:
                assert row["batch_id"] == "my_batch"

    def test_step_values_are_correct(self, tmp_path: Path):
        raw_rows = [_make_paysim_csv_row(i) for i in range(5)]
        csv_path = _make_csv(tmp_path, raw_rows)

        all_rows = [row for chunk in read_paysim_chunks(csv_path) for row in chunk]
        steps = [r["step"] for r in all_rows]
        assert steps == ["0", "1", "2", "3", "4"]

    def test_amount_values_are_strings(self, tmp_path: Path):
        """Amount must be kept as string (cast happens later in Spark)."""
        raw_rows = [_make_paysim_csv_row(0)]
        csv_path = _make_csv(tmp_path, raw_rows)

        chunks = list(read_paysim_chunks(csv_path))
        amount = chunks[0][0]["amount"]
        assert isinstance(amount, str)

    def test_file_not_found_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            list(read_paysim_chunks(tmp_path / "nonexistent.csv"))

    def test_single_chunk_when_rows_less_than_chunk_size(self, tmp_path: Path):
        raw_rows = [_make_paysim_csv_row(i) for i in range(5)]
        csv_path = _make_csv(tmp_path, raw_rows)

        chunks = list(read_paysim_chunks(csv_path, chunk_size=100))
        assert len(chunks) == 1
        assert len(chunks[0]) == 5
