"""
test_generators.py — Unit tests for M2 synthetic generators.
"""
from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.generators.enrich_paysim import (
    MONEY_COLS,
    enrich_chunk,
)
from src.generators.fx_rates import (
    ABSENT_PAIRS,
    generate_fx_rates,
    write_fx_rates_parquet,
)
from src.generators.generate_settlements import (
    generate_settlements,
)
from src.generators.inject_anomalies import (
    build_manifest,
)


class TestFxRates:
    def test_generate_fx_rates_keys_and_types(self):
        rates = generate_fx_rates(seed=42)
        assert len(rates) > 0
        sample = rates[0]
        assert "currency_pair" in sample
        assert "rate" in sample
        assert "effective_timestamp" in sample
        # Rate must be stored as Decimal string (not float)
        assert isinstance(sample["rate"], str)
        # Should parse as Decimal with 6 decimal places
        d_rate = Decimal(sample["rate"])
        assert d_rate.as_tuple().exponent == -6

    def test_absent_pairs_excluded(self):
        rates = generate_fx_rates(seed=42)
        pairs = {r["currency_pair"] for r in rates}
        for absent in ABSENT_PAIRS:
            assert absent not in pairs, f"Absent pair {absent} found in generated rates!"

    def test_write_fx_rates_parquet(self, tmp_path: Path):
        rates = generate_fx_rates(seed=42)
        out_file = tmp_path / "fx_rates.parquet"
        write_fx_rates_parquet(rates, out_file)
        assert out_file.exists()


class TestEnrichPaysim:
    @pytest.fixture
    def sample_chunk(self) -> pd.DataFrame:
        data = {
            "step": [1, 2, 3],
            "type": ["PAYMENT", "TRANSFER", "CASH_OUT"],
            "amount": [100.5, 200.75, 50.0],
            "nameOrig": ["C1", "C2", "C3"],
            "oldbalanceOrg": [500.0, 1000.0, 250.0],
            "newbalanceOrig": [399.5, 799.25, 200.0],
            "nameDest": ["M1", "C4", "M2"],
            "oldbalanceDest": [0.0, 50.0, 10.0],
            "newbalanceDest": [0.0, 250.75, 60.0],
            "isFraud": [0, 0, 1],
            "isFlaggedFraud": [0, 0, 0],
        }
        return pd.DataFrame(data)

    def test_enrich_chunk_columns(self, sample_chunk: pd.DataFrame):
        rng = np.random.default_rng(42)
        base_date = datetime(2023, 1, 1, 0, 0, 0, tzinfo=UTC)
        enriched = enrich_chunk(sample_chunk, start_idx=0, rng=rng, base_date=base_date)

        assert "transaction_id" in enriched.columns
        assert "currency" in enriched.columns
        assert "fx_pair" in enriched.columns
        assert "transaction_timestamp" in enriched.columns

        # Verify money columns are strings (not float)
        for col in MONEY_COLS:
            assert all(isinstance(v, str) for v in enriched[col])
            # Each should parse cleanly as Decimal with 2 decimal places
            for v in enriched[col]:
                d = Decimal(v)
                assert d.as_tuple().exponent == -2

    def test_timestamp_calculation(self, sample_chunk: pd.DataFrame):
        rng = np.random.default_rng(42)
        base_date = datetime(2023, 1, 1, 0, 0, 0, tzinfo=UTC)
        enriched = enrich_chunk(sample_chunk, start_idx=0, rng=rng, base_date=base_date)
        # Step 1 should be 2023-01-01T01:00:00+00:00
        assert enriched.iloc[0]["transaction_timestamp"] == "2023-01-01T01:00:00+00:00"


class TestSettlementsAndAnomalies:
    def test_generate_settlements(self, tmp_path: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        txn_df = pd.DataFrame({
            "transaction_id": ["tx-1", "tx-2", "tx-3", "tx-4"],
            "currency": ["USD", "EUR", "GBP", "USD"],
            "amount": ["100.00", "250.50", "300.00", "50.00"],
            "transaction_timestamp": [
                "2023-01-01T01:00:00+00:00",
                "2023-01-01T02:00:00+00:00",
                "2023-01-01T03:00:00+00:00",
                "2023-01-01T04:00:00+00:00",
            ],
        })
        in_file = tmp_path / "txns.parquet"
        out_file = tmp_path / "settlements.parquet"
        pq.write_table(pa.Table.from_pandas(txn_df), in_file)

        generate_settlements(
            input_parquet=in_file,
            output_parquet=out_file,
            seed=42,
            missing_settlement_rate=0.25,
            duplicate_settlement_rate=0.25,
            amount_mismatch_rate=0.25,
        )
        assert out_file.exists()
        stl_df = pq.read_table(out_file).to_pandas()
        assert len(stl_df) > 0
        assert "settlement_id" in stl_df.columns
        assert "settled_amount" in stl_df.columns

    def test_build_manifest(self):
        records = [
            {
                "anomaly_id": "anom-1",
                "transaction_id": "tx-1",
                "settlement_id": "stl-1",
                "anomaly_type": "AMOUNT_MISMATCH",
                "injection_batch_id": "batch_001",
                "expected_detection": True,
                "anomaly_details": "{\"diff\": 5.0}",
            }
        ]
        manifest_df = build_manifest(records)
        assert len(manifest_df) == 1
        assert "anomaly_id" in manifest_df.columns
        assert manifest_df.iloc[0]["anomaly_type"] == "AMOUNT_MISMATCH"
