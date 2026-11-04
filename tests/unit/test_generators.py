"""
test_generators.py — Unit tests for M2 synthetic generators.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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
    inject_anomalies,
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

    def test_usd_usd_identity_generated(self):
        rates = generate_fx_rates(seed=42)
        usd_rates = [r for r in rates if r["currency_pair"] == "USD/USD"]
        assert len(usd_rates) > 0
        for r in usd_rates:
            assert r["rate"] == "1.000000"

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

    def test_manifest_data_consistency(self, tmp_path: Path):
        """Verify that every manifest entry is directly observable in the mutated parquet data."""
        # 1. Create a synthetic enriched transactions parquet
        n = 50
        txn_data = {
            "transaction_id": [f"tx-{i}" for i in range(n)],
            "currency": ["USD" if i % 2 == 0 else "EUR" for i in range(n)],
            "amount": [f"{(i + 1) * 10}.00" for i in range(n)],
            "fx_pair": ["USD/USD" if i % 2 == 0 else "EUR/USD" for i in range(n)],
            "transaction_timestamp": [f"2023-01-01T{(i % 24):02d}:00:00+00:00" for i in range(n)],
            "step": [i % 24 for i in range(n)],
            "type": ["PAYMENT" for _ in range(n)],
            "nameOrig": [f"C{i}" for i in range(n)],
            "oldbalanceOrg": ["1000.00" for _ in range(n)],
            "newbalanceOrig": ["900.00" for _ in range(n)],
            "nameDest": [f"M{i}" for i in range(n)],
            "oldbalanceDest": ["0.00" for _ in range(n)],
            "newbalanceDest": ["100.00" for _ in range(n)],
            "isFraud": [0 for _ in range(n)],
            "isFlaggedFraud": [0 for _ in range(n)],
        }
        txn_file = tmp_path / "enriched.parquet"
        stl_file = tmp_path / "settlements.parquet"
        manifest_file = tmp_path / "manifest.parquet"
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(txn_data)), txn_file)

        # 2. Generate settlements with anomalies
        prior_anomalies = generate_settlements(
            input_parquet=txn_file,
            output_parquet=stl_file,
            seed=42,
            missing_settlement_rate=0.1,
            duplicate_settlement_rate=0.1,
            amount_mismatch_rate=0.1,
        )

        # 3. Inject remaining anomalies and mutate the datasets
        inject_anomalies(
            enriched_parquet=txn_file,
            settlements_parquet=stl_file,
            output_manifest=manifest_file,
            seed=42,
            prior_anomalies=prior_anomalies,
            duplicate_transaction_rate=0.1,
            invalid_record_rate=0.1,
            orphan_settlement_rate=0.1,
            delayed_settlement_rate=0.1,
            missing_fx_rate=0.1,
        )

        # 4. Read back mutated datasets and manifest
        updated_txns = pq.read_table(txn_file).to_pandas()
        updated_stls = pq.read_table(stl_file).to_pandas()
        manifest = pq.read_table(manifest_file).to_pandas()

        assert len(manifest) > 0

        # Build lookup indices for consistency verification
        txn_ids_in_txns = set(updated_txns["transaction_id"].dropna().tolist())
        stl_ids_in_stls = set(updated_stls["settlement_id"].tolist())

        # 5. Verify every manifest row is observable in the data
        for _, row in manifest.iterrows():
            anom_type = row["anomaly_type"]
            tid = row["transaction_id"]
            sid = row["settlement_id"]
            details = json.loads(row["anomaly_details"]) if isinstance(row["anomaly_details"], str) else row["anomaly_details"]

            if anom_type == "DUPLICATE_TXN":
                count = len(updated_txns[updated_txns["transaction_id"] == tid])
                assert count >= 2, f"DUPLICATE_TXN {tid} not duplicated in transactions (count={count})"

            elif anom_type == "INVALID_RECORD":
                # Either empty/null transaction_id or negative amount
                matching_corrupt = updated_txns[
                    (updated_txns["transaction_id"] == "") |
                    (updated_txns["amount"] == "-1.00")
                ]
                assert len(matching_corrupt) > 0, "INVALID_RECORD not found in transactions"

            elif anom_type == "MISSING_FX_RATE":
                matching = updated_txns[updated_txns["transaction_id"] == tid]
                assert len(matching) > 0
                ts = matching.iloc[0]["transaction_timestamp"]
                assert "2022" in str(ts), f"MISSING_FX_RATE {tid} timestamp not pre-dated ({ts})"

            elif anom_type == "ORPHAN_SETTLEMENT":
                assert sid in stl_ids_in_stls, f"ORPHAN_SETTLEMENT {sid} not present in settlements"
                assert tid not in txn_ids_in_txns, f"ORPHAN_SETTLEMENT txn_id {tid} unexpectedly found in transactions"

            elif anom_type == "DELAYED_SETTLEMENT":
                assert sid in stl_ids_in_stls
                matching_stl = updated_stls[updated_stls["settlement_id"] == sid].iloc[0]
                expected_ts = details["delayed_settlement_timestamp"]
                assert matching_stl["settlement_timestamp"] == expected_ts

            elif anom_type == "MISSING_SETTLEMENT":
                matching = updated_stls[updated_stls["transaction_id"] == tid]
                assert len(matching) == 0, f"MISSING_SETTLEMENT {tid} unexpectedly found in settlements"

            elif anom_type == "DUPLICATE_SETTLEMENT":
                matching = updated_stls[updated_stls["transaction_id"] == tid]
                assert len(matching) >= 2, f"DUPLICATE_SETTLEMENT {tid} has {len(matching)} settlements (expected >=2)"

            elif anom_type == "AMOUNT_MISMATCH":
                matching_stl = updated_stls[updated_stls["transaction_id"] == tid]
                assert len(matching_stl) > 0
                assert details["original_amount"] != details["settled_amount"]
                assert matching_stl.iloc[0]["settled_amount"] == details["settled_amount"]
