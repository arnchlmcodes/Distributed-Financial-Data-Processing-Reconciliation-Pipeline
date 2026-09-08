"""
enrich_paysim.py — Enrich PaySim CSV with synthetic fields.

Processes the CSV in chunks (default 50 000 rows) and adds:
- transaction_id  : UUID4 (deterministic per seed + row index)
- currency        : random choice from settings.currencies
- fx_pair         : f"{currency}/USD"
- transaction_timestamp : base_date + timedelta(hours=step)
- money columns   : converted from float CSV → Decimal string (2 dp)

Writes:
  data/processed/enriched_sample.parquet  (first 50 K rows, always)
  data/processed/enriched_transactions.parquet  (all rows, only with --full)
"""
from __future__ import annotations

import argparse
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MONEY_COLS: list[str] = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

TWO_DP = Decimal("0.01")

# Pyarrow schema for the enriched table
_ENRICHED_SCHEMA = pa.schema(
    [
        pa.field("step", pa.int64()),
        pa.field("type", pa.string()),
        pa.field("amount", pa.string()),
        pa.field("nameOrig", pa.string()),
        pa.field("oldbalanceOrg", pa.string()),
        pa.field("newbalanceOrig", pa.string()),
        pa.field("nameDest", pa.string()),
        pa.field("oldbalanceDest", pa.string()),
        pa.field("newbalanceDest", pa.string()),
        pa.field("isFraud", pa.int64()),
        pa.field("isFlaggedFraud", pa.int64()),
        pa.field("transaction_id", pa.string()),
        pa.field("currency", pa.string()),
        pa.field("fx_pair", pa.string()),
        pa.field("transaction_timestamp", pa.string()),
    ]
)


# ---------------------------------------------------------------------------
# Core transformation
# ---------------------------------------------------------------------------

def _float_to_decimal_str(val: float) -> str:
    """Convert a float money value to a 2-dp Decimal string."""
    return str(Decimal(str(val)).quantize(TWO_DP, rounding=ROUND_HALF_UP))


def _make_transaction_id(seed: int, row_index: int) -> str:
    """Generate a deterministic UUID4 from seed and row index."""
    # Use UUID5 with a fixed namespace + seed + index for determinism
    name = f"{seed}:{row_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_OID, name))


def enrich_chunk(
    chunk_df: pd.DataFrame,
    start_idx: int,
    rng: np.random.Generator,
    base_date: datetime,
    currencies: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Enrich a single DataFrame chunk with synthetic fields.

    Parameters
    ----------
    chunk_df : pd.DataFrame
        Raw PaySim chunk (columns as per CSV header).
    start_idx : int
        Absolute row index of the first row in this chunk.
    rng : np.random.Generator
        Seeded NumPy RNG (caller controls state across chunks).
    base_date : datetime
        UTC datetime for step 0 (base_date + step*1h = transaction timestamp).
    currencies : list[str] | None
        Currency list to sample from; defaults to settings.currencies.
    seed : int
        Seed used for deterministic transaction_id generation.

    Returns
    -------
    pd.DataFrame
        Enriched chunk with all original columns plus the new synthetic ones.
    """
    if currencies is None:
        from src.config import settings  # lazy import avoids circular deps at test time
        currencies = settings.currencies

    n = len(chunk_df)
    indices = range(start_idx, start_idx + n)

    # --- transaction_id (deterministic UUID5 from seed + absolute row index) ---
    chunk_df = chunk_df.copy()
    chunk_df["transaction_id"] = [_make_transaction_id(seed, i) for i in indices]

    # --- currency (seeded random choice) ---
    chosen_currencies: list[str] = currencies  # keep reference
    currency_indices = rng.integers(0, len(chosen_currencies), size=n)
    chunk_df["currency"] = [chosen_currencies[ci] for ci in currency_indices]

    # --- fx_pair ---
    chunk_df["fx_pair"] = chunk_df["currency"].apply(lambda c: f"{c}/USD")

    # --- transaction_timestamp ---
    chunk_df["transaction_timestamp"] = chunk_df["step"].apply(
        lambda s: (base_date + timedelta(hours=int(s))).isoformat()
    )

    # --- money columns: float → Decimal string (2 dp) ---
    for col in MONEY_COLS:
        if col in chunk_df.columns:
            chunk_df[col] = chunk_df[col].apply(_float_to_decimal_str)

    # Ensure integer columns stay as int64
    for col in ("step", "isFraud", "isFlaggedFraud"):
        if col in chunk_df.columns:
            chunk_df[col] = chunk_df[col].astype("int64")

    return chunk_df


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write enriched DataFrame to Parquet using the canonical schema."""
    table = pa.Table.from_pandas(df, schema=_ENRICHED_SCHEMA, preserve_index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    logger.info("Wrote %d rows → %s", len(df), path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(full: bool = False) -> None:
    """
    Enrich PaySim CSV.

    Parameters
    ----------
    full : bool
        If True, process all rows and write enriched_transactions.parquet.
        If False, process only the first chunk (50 K rows) as a sample.
    """
    from src.config import settings

    raw_csv = settings.raw_dir / "paysim.csv"
    sample_out = settings.processed_dir / "enriched_sample.parquet"
    full_out = settings.processed_dir / "enriched_transactions.parquet"

    base_date = datetime.fromisoformat(settings.base_date).replace(tzinfo=UTC)
    chunk_size = settings.batch_size
    rng = np.random.default_rng(settings.random_seed)

    logger.info(
        "Starting enrichment: full=%s, chunk_size=%d, base_date=%s",
        full,
        chunk_size,
        base_date.isoformat(),
    )

    sample_written = False
    total_rows = 0

    writer: pq.ParquetWriter | None = None
    full_path = full_out if full else None

    try:
        reader = pd.read_csv(raw_csv, chunksize=chunk_size, dtype=str)
        for chunk_num, raw_chunk in enumerate(reader):
            # Cast numeric columns before passing to enrich_chunk
            for col in MONEY_COLS + ["step", "isFraud", "isFlaggedFraud"]:
                if col in raw_chunk.columns:
                    try:
                        raw_chunk[col] = pd.to_numeric(raw_chunk[col])
                    except (ValueError, TypeError):
                        pass  # will be handled inside enrich_chunk

            start_idx = chunk_num * chunk_size
            enriched = enrich_chunk(
                raw_chunk,
                start_idx=start_idx,
                rng=rng,
                base_date=base_date,
                currencies=settings.currencies,
                seed=settings.random_seed,
            )

            # Write sample from first chunk
            if not sample_written:
                _write_parquet(enriched, sample_out)
                sample_written = True
                logger.info("Sample written: %d rows → %s", len(enriched), sample_out)
                if not full:
                    logger.info("Sample-only mode: stopping after first chunk.")
                    total_rows = len(enriched)
                    break

            # Full mode: stream to parquet writer
            if full:
                table = pa.Table.from_pandas(
                    enriched, schema=_ENRICHED_SCHEMA, preserve_index=False
                )
                if writer is None:
                    writer = pq.ParquetWriter(full_path, schema=_ENRICHED_SCHEMA)
                writer.write_table(table)
                total_rows += len(enriched)

                if chunk_num % 10 == 0:
                    logger.info("Processed chunk %d (%d rows so far)", chunk_num, total_rows)
    finally:
        if writer is not None:
            writer.close()

    if full:
        logger.info("Full enrichment complete: %d total rows → %s", total_rows, full_path)
    print(f"Enrichment done. Rows processed: {total_rows}. Sample: {sample_out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Enrich PaySim CSV with synthetic fields.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Process all rows and write enriched_transactions.parquet",
    )
    args = parser.parse_args()
    run(full=args.full)
