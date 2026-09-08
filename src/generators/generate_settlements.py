"""
generate_settlements.py — Generate settlement records for enriched transactions.

For each transaction one of four outcomes is chosen (seeded RNG):
  - MISSING_SETTLEMENT  (2%): no settlement row created
  - DUPLICATE_SETTLEMENT (1%): two settlement rows created
  - AMOUNT_MISMATCH      (1%): settlement amount differs 1-5% from transaction
  - Normal               (rest): single correct settlement

All injected anomalies are tracked in a list returned alongside the
settlements DataFrame so that inject_anomalies.py can build the manifest.

Entry point: ``generate_settlements(input_parquet, output_parquet, seed)``
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

TWO_DP = Decimal("0.01")

_SETTLEMENT_SCHEMA = pa.schema(
    [
        pa.field("settlement_id", pa.string()),
        pa.field("transaction_id", pa.string()),
        pa.field("settled_amount", pa.string()),
        pa.field("settlement_currency", pa.string()),
        pa.field("settlement_timestamp", pa.string()),
        pa.field("status", pa.string()),
    ]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decimal_str(val: str | float | Decimal) -> str:
    """Normalise a value to 2-dp Decimal string."""
    return str(Decimal(str(val)).quantize(TWO_DP, rounding=ROUND_HALF_UP))


def _mismatch_amount(amount_str: str, rng: np.random.Generator) -> str:
    """Return amount shifted by 1-5% (positive or negative)."""
    base = Decimal(amount_str)
    pct = Decimal(str(round(float(rng.uniform(0.01, 0.05)), 6)))
    sign = Decimal(1) if rng.integers(0, 2) == 0 else Decimal(-1)
    adjusted = base + base * pct * sign
    # Amount must be positive
    if adjusted <= Decimal(0):
        adjusted = base * Decimal("0.99")
    return str(adjusted.quantize(TWO_DP, rounding=ROUND_HALF_UP))


def _settlement_timestamp(txn_ts: str, rng: np.random.Generator) -> str:
    """Return txn_timestamp + random 1-48 hours (UTC ISO8601)."""
    base = datetime.fromisoformat(txn_ts)
    if base.tzinfo is None:
        base = base.replace(tzinfo=UTC)
    delay_hours = int(rng.integers(1, 49))
    settled_at = base + timedelta(hours=delay_hours)
    return settled_at.isoformat()


def _make_settlement_row(
    txn_id: str,
    amount: str,
    currency: str,
    txn_ts: str,
    rng: np.random.Generator,
    status: str = "COMPLETED",
) -> dict:
    """Build a single settlement dict."""
    return {
        "settlement_id": str(uuid.uuid4()),
        "transaction_id": txn_id,
        "settled_amount": amount,
        "settlement_currency": currency,
        "settlement_timestamp": _settlement_timestamp(txn_ts, rng),
        "status": status,
    }


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_settlements(
    input_parquet: Path,
    output_parquet: Path,
    seed: int = 42,
    missing_settlement_rate: float = 0.02,
    duplicate_settlement_rate: float = 0.01,
    amount_mismatch_rate: float = 0.01,
    injection_batch_id: str | None = None,
) -> list[dict]:
    """
    Generate settlement records from enriched transactions.

    Parameters
    ----------
    input_parquet : Path
        Path to the enriched_sample.parquet (or enriched_transactions.parquet).
    output_parquet : Path
        Destination for settlements_sample.parquet.
    seed : int
        RNG seed.
    missing_settlement_rate : float
        Fraction of transactions that will have no settlement.
    duplicate_settlement_rate : float
        Fraction of transactions that will have duplicate settlements.
    amount_mismatch_rate : float
        Fraction of transactions with a settlement amount mismatch.
    injection_batch_id : str | None
        UUID string for this injection batch; auto-generated if None.

    Returns
    -------
    list[dict]
        Anomaly records (each dict suitable for the manifest).
    """
    if injection_batch_id is None:
        injection_batch_id = str(uuid.uuid4())

    rng = np.random.default_rng(seed)
    anomaly_records: list[dict] = []
    settlement_rows: list[dict] = []

    df = pq.read_table(input_parquet).to_pandas()
    logger.info("Loaded %d transactions from %s", len(df), input_parquet)

    # Pre-draw random floats for efficiency
    rolls = rng.random(size=len(df))

    for i, (_, row) in enumerate(df.iterrows()):
        txn_id = str(row["transaction_id"])
        amount = str(row["amount"])
        currency = str(row["currency"])
        txn_ts = str(row["transaction_timestamp"])
        roll = float(rolls[i])

        # Order of evaluation: MISSING first, then DUPLICATE, then MISMATCH
        if roll < missing_settlement_rate:
            # No settlement row — just record the anomaly
            anomaly_records.append(
                {
                    "anomaly_id": str(uuid.uuid4()),
                    "transaction_id": txn_id,
                    "settlement_id": None,
                    "anomaly_type": "MISSING_SETTLEMENT",
                    "injection_batch_id": injection_batch_id,
                    "expected_detection": True,
                    "anomaly_details": json.dumps({"reason": "no settlement generated"}),
                }
            )

        elif roll < missing_settlement_rate + duplicate_settlement_rate:
            # Two settlement rows — anomaly
            s1 = _make_settlement_row(txn_id, amount, currency, txn_ts, rng)
            s2 = _make_settlement_row(txn_id, amount, currency, txn_ts, rng)
            settlement_rows.extend([s1, s2])
            anomaly_records.append(
                {
                    "anomaly_id": str(uuid.uuid4()),
                    "transaction_id": txn_id,
                    "settlement_id": s1["settlement_id"],
                    "anomaly_type": "DUPLICATE_SETTLEMENT",
                    "injection_batch_id": injection_batch_id,
                    "expected_detection": True,
                    "anomaly_details": json.dumps(
                        {
                            "settlement_ids": [s1["settlement_id"], s2["settlement_id"]],
                            "reason": "duplicate settlement injected",
                        }
                    ),
                }
            )

        elif roll < missing_settlement_rate + duplicate_settlement_rate + amount_mismatch_rate:
            # Settlement with mismatched amount
            mismatched = _mismatch_amount(amount, rng)
            s = _make_settlement_row(txn_id, mismatched, currency, txn_ts, rng)
            settlement_rows.append(s)
            anomaly_records.append(
                {
                    "anomaly_id": str(uuid.uuid4()),
                    "transaction_id": txn_id,
                    "settlement_id": s["settlement_id"],
                    "anomaly_type": "AMOUNT_MISMATCH",
                    "injection_batch_id": injection_batch_id,
                    "expected_detection": True,
                    "anomaly_details": json.dumps(
                        {
                            "original_amount": amount,
                            "settled_amount": mismatched,
                        }
                    ),
                }
            )

        else:
            # Normal settlement
            s = _make_settlement_row(txn_id, amount, currency, txn_ts, rng)
            settlement_rows.append(s)

    # Write settlements
    settlements_df = pd.DataFrame(settlement_rows)
    if settlements_df.empty:
        settlements_df = pd.DataFrame(columns=[f.name for f in _SETTLEMENT_SCHEMA])

    table = pa.Table.from_pandas(settlements_df, schema=_SETTLEMENT_SCHEMA, preserve_index=False)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_parquet)

    logger.info(
        "Settlements written: %d rows → %s | anomalies injected: %d",
        len(settlement_rows),
        output_parquet,
        len(anomaly_records),
    )
    return anomaly_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from src.config import settings

    inp = settings.processed_dir / "enriched_sample.parquet"
    out = settings.processed_dir / "settlements_sample.parquet"
    anomalies = generate_settlements(
        inp,
        out,
        seed=settings.random_seed,
        missing_settlement_rate=settings.missing_settlement_rate,
        duplicate_settlement_rate=settings.duplicate_settlement_rate,
        amount_mismatch_rate=settings.amount_mismatch_rate,
    )
    print(f"Settlements generated. Anomalies in this batch: {len(anomalies)}")
