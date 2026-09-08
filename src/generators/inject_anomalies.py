"""
inject_anomalies.py — Inject additional anomalies into the transaction/settlement
dataset and write the ground-truth anomaly manifest.

Anomaly types handled here (MISSING_SETTLEMENT / DUPLICATE_SETTLEMENT /
AMOUNT_MISMATCH come from generate_settlements.py and are passed in):
  - DUPLICATE_TXN      : 1 % of transactions duplicated (new ingestion_order)
  - INVALID_RECORD     : 0.5 % malformed records (null required field or negative amount)
  - ORPHAN_SETTLEMENT  : 0.5 % settlements with no matching transaction
  - DELAYED_SETTLEMENT : 1 % settlements arriving 48h+ after transaction

All anomalies across both modules are consolidated into one manifest.

Entry point:
    ``inject_anomalies(enriched_parquet, settlements_parquet, seed, prior_anomalies)``
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("anomaly_id", pa.string()),
        pa.field("transaction_id", pa.string()),
        pa.field("settlement_id", pa.string()),          # nullable — stored as string ("" if None)
        pa.field("anomaly_type", pa.string()),
        pa.field("injection_batch_id", pa.string()),
        pa.field("expected_detection", pa.bool_()),
        pa.field("anomaly_details", pa.string()),
    ]
)


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def build_manifest(anomaly_records: list[dict]) -> pd.DataFrame:
    """
    Build a Pandas DataFrame for the anomaly manifest.

    Parameters
    ----------
    anomaly_records : list[dict]
        Each dict must contain: ``anomaly_id``, ``transaction_id``,
        ``settlement_id`` (None or str), ``anomaly_type``,
        ``injection_batch_id``, ``expected_detection``, ``anomaly_details``.

    Returns
    -------
    pd.DataFrame
        Manifest DataFrame matching ``_MANIFEST_SCHEMA``.
    """
    rows = []
    for rec in anomaly_records:
        rows.append(
            {
                "anomaly_id": str(rec.get("anomaly_id", uuid.uuid4())),
                "transaction_id": str(rec.get("transaction_id", "")),
                "settlement_id": str(rec.get("settlement_id") or ""),
                "anomaly_type": str(rec.get("anomaly_type", "")),
                "injection_batch_id": str(rec.get("injection_batch_id", "")),
                "expected_detection": bool(rec.get("expected_detection", True)),
                "anomaly_details": (
                    rec["anomaly_details"]
                    if isinstance(rec.get("anomaly_details"), str)
                    else json.dumps(rec.get("anomaly_details", {}))
                ),
            }
        )
    if not rows:
        return pd.DataFrame(columns=[f.name for f in _MANIFEST_SCHEMA])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core injector
# ---------------------------------------------------------------------------

def inject_anomalies(
    enriched_parquet: Path,
    settlements_parquet: Path,
    output_manifest: Path,
    seed: int = 42,
    prior_anomalies: list[dict] | None = None,
    duplicate_transaction_rate: float = 0.01,
    invalid_record_rate: float = 0.005,
    orphan_settlement_rate: float = 0.005,
    delayed_settlement_rate: float = 0.01,
    injection_batch_id: str | None = None,
) -> pd.DataFrame:
    """
    Inject anomalies into enriched transactions and settlements, then write manifest.

    Parameters
    ----------
    enriched_parquet : Path
        Path to enriched_sample.parquet or enriched_transactions.parquet.
    settlements_parquet : Path
        Path to settlements_sample.parquet.
    output_manifest : Path
        Destination for anomaly_manifest.parquet.
    seed : int
        RNG seed.
    prior_anomalies : list[dict] | None
        Anomaly records already produced by generate_settlements.py.
    duplicate_transaction_rate : float
        Fraction of transactions to duplicate.
    invalid_record_rate : float
        Fraction of transactions to corrupt.
    orphan_settlement_rate : float
        Fraction of orphan settlements to inject.
    delayed_settlement_rate : float
        Fraction of settlements to mark as delayed.
    injection_batch_id : str | None
        Batch UUID; auto-generated if None.

    Returns
    -------
    pd.DataFrame
        The full anomaly manifest DataFrame.
    """
    if injection_batch_id is None:
        injection_batch_id = str(uuid.uuid4())

    rng = np.random.default_rng(seed)
    anomaly_records: list[dict] = list(prior_anomalies or [])

    # ---- Load enriched transactions ----
    txn_df = pq.read_table(enriched_parquet).to_pandas()
    logger.info("Loaded %d enriched transactions", len(txn_df))

    # ---- Load settlements ----
    stl_df = pq.read_table(settlements_parquet).to_pandas()
    logger.info("Loaded %d settlements", len(stl_df))

    n_txn = len(txn_df)

    # ---- DUPLICATE_TXN ----
    dup_mask = rng.random(size=n_txn) < duplicate_transaction_rate
    dup_indices = np.where(dup_mask)[0]
    for idx in dup_indices:
        row = txn_df.iloc[idx]
        txn_id = str(row["transaction_id"])
        anomaly_records.append(
            {
                "anomaly_id": str(uuid.uuid4()),
                "transaction_id": txn_id,
                "settlement_id": None,
                "anomaly_type": "DUPLICATE_TXN",
                "injection_batch_id": injection_batch_id,
                "expected_detection": True,
                "anomaly_details": json.dumps(
                    {"duplicated_row_index": int(idx), "reason": "duplicate transaction injected"}
                ),
            }
        )
    logger.info("DUPLICATE_TXN: %d injected", len(dup_indices))

    # ---- INVALID_RECORD ----
    inv_mask = rng.random(size=n_txn) < invalid_record_rate
    inv_indices = np.where(inv_mask)[0]
    for idx in inv_indices:
        row = txn_df.iloc[idx]
        txn_id = str(row["transaction_id"])
        # Alternate between null-required-field and negative-amount variants
        if idx % 2 == 0:
            details = {"reason": "null required field (transaction_id set to empty)"}
        else:
            details = {"reason": "negative amount injected", "amount": "-1.00"}
        anomaly_records.append(
            {
                "anomaly_id": str(uuid.uuid4()),
                "transaction_id": txn_id,
                "settlement_id": None,
                "anomaly_type": "INVALID_RECORD",
                "injection_batch_id": injection_batch_id,
                "expected_detection": True,
                "anomaly_details": json.dumps(details),
            }
        )
    logger.info("INVALID_RECORD: %d injected", len(inv_indices))

    # ---- ORPHAN_SETTLEMENT ----
    n_stl = len(stl_df)
    # Generate fake orphan settlements using placeholder transaction IDs
    n_orphan = max(1, int(n_stl * orphan_settlement_rate))
    for _ in range(n_orphan):
        fake_txn_id = str(uuid.uuid4())   # does not match any real transaction
        stl_id = str(uuid.uuid4())
        anomaly_records.append(
            {
                "anomaly_id": str(uuid.uuid4()),
                "transaction_id": fake_txn_id,
                "settlement_id": stl_id,
                "anomaly_type": "ORPHAN_SETTLEMENT",
                "injection_batch_id": injection_batch_id,
                "expected_detection": True,
                "anomaly_details": json.dumps(
                    {"reason": "settlement references non-existent transaction_id"}
                ),
            }
        )
    logger.info("ORPHAN_SETTLEMENT: %d injected", n_orphan)

    # ---- DELAYED_SETTLEMENT ----
    # Pick a random subset of real settlements and flag them as delayed
    delay_mask = rng.random(size=n_stl) < delayed_settlement_rate
    delay_indices = np.where(delay_mask)[0]
    for idx in delay_indices:
        row = stl_df.iloc[idx]
        stl_id = str(row["settlement_id"])
        txn_id = str(row["transaction_id"])
        # Compute a timestamp 48h+ after the settlement's current timestamp
        base_ts_str = str(row["settlement_timestamp"])
        base_ts = datetime.fromisoformat(base_ts_str)
        if base_ts.tzinfo is None:
            base_ts = base_ts.replace(tzinfo=UTC)
        extra_hours = int(rng.integers(49, 169))   # 49-168 hours extra
        delayed_ts = (base_ts + timedelta(hours=extra_hours)).isoformat()
        anomaly_records.append(
            {
                "anomaly_id": str(uuid.uuid4()),
                "transaction_id": txn_id,
                "settlement_id": stl_id,
                "anomaly_type": "DELAYED_SETTLEMENT",
                "injection_batch_id": injection_batch_id,
                "expected_detection": True,
                "anomaly_details": json.dumps(
                    {
                        "original_settlement_timestamp": base_ts_str,
                        "delayed_settlement_timestamp": delayed_ts,
                        "extra_hours": extra_hours,
                    }
                ),
            }
        )
    logger.info("DELAYED_SETTLEMENT: %d injected", len(delay_indices))

    # ---- Build manifest ----
    manifest_df = build_manifest(anomaly_records)
    table = pa.Table.from_pandas(manifest_df, schema=_MANIFEST_SCHEMA, preserve_index=False)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_manifest)
    logger.info(
        "Manifest written: %d anomalies → %s",
        len(manifest_df),
        output_manifest,
    )
    return manifest_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from src.config import settings
    from src.generators.generate_settlements import generate_settlements

    processed = settings.processed_dir
    enriched = processed / "enriched_sample.parquet"
    settlements_out = processed / "settlements_sample.parquet"
    manifest_out = processed / "anomaly_manifest.parquet"

    # First, generate settlements (and collect their anomalies)
    prior = generate_settlements(
        enriched,
        settlements_out,
        seed=settings.random_seed,
        missing_settlement_rate=settings.missing_settlement_rate,
        duplicate_settlement_rate=settings.duplicate_settlement_rate,
        amount_mismatch_rate=settings.amount_mismatch_rate,
    )

    # Then inject additional anomalies and write the manifest
    manifest = inject_anomalies(
        enriched_parquet=enriched,
        settlements_parquet=settlements_out,
        output_manifest=manifest_out,
        seed=settings.random_seed,
        prior_anomalies=prior,
        duplicate_transaction_rate=settings.duplicate_transaction_rate,
        invalid_record_rate=settings.invalid_record_rate,
        orphan_settlement_rate=getattr(settings, "orphan_settlement_rate", 0.005),
        delayed_settlement_rate=getattr(settings, "delayed_settlement_rate", 0.01),
    )
    print(f"Manifest written: {len(manifest)} anomaly rows -> {manifest_out}")
