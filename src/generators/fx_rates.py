"""
fx_rates.py — Generate synthetic FX rate table.

Currencies: USD, EUR, GBP, INR, NGN, KES
Pairs: all combinations except NGN/KES and KES/NGN (intentionally absent
       to trigger MISSING_FX_RATE anomalies downstream).
Rates are stored at 6 decimal places as Decimal-serialised strings.
One rate per pair per 24-hour step (743 steps ≈ 30 days).
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CURRENCIES: list[str] = ["USD", "EUR", "GBP", "INR", "NGN", "KES"]
ABSENT_PAIRS: frozenset[str] = frozenset({"NGN/KES", "KES/NGN"})
DATASET_STEPS: int = 743          # 743 one-hour steps → 743 hours ≈ 31 days
DAILY_STEPS: int = 24             # one rate per 24 hours
BASE_DATE: datetime = datetime(2023, 1, 1, 0, 0, 0, tzinfo=UTC)

# Approximate mid-market USD base rates (mean, daily_vol %)
# All pairs are derived as USD/X or X/USD from these anchors.
_USD_BASE: dict[str, tuple[float, float]] = {
    "EUR": (0.905, 0.003),
    "GBP": (0.785, 0.003),
    "INR": (83.5, 0.005),
    "NGN": (760.0, 0.008),
    "KES": (135.0, 0.006),
}

SIX_DP: Decimal = Decimal("0.000001")


def _rate_to_str(rate: float) -> str:
    """Convert a float rate to a 6-decimal-place Decimal string."""
    return str(Decimal(str(rate)).quantize(SIX_DP, rounding=ROUND_HALF_UP))


def generate_fx_rates(seed: int = 42) -> list[dict]:
    """
    Generate FX rate rows for all currency permutations (except absent pairs).

    Each pair gets one rate per 24-hour window for the full dataset duration.
    Rates are seeded random-walk values stored as Decimal strings (6 dp).

    Parameters
    ----------
    seed : int
        Seed for the NumPy default RNG.

    Returns
    -------
    list[dict]
        Each dict has keys: ``currency_pair``, ``rate``, ``effective_timestamp``.
    """
    rng = np.random.default_rng(seed)

    # Build daily timestamps: step 0, 24, 48, … up to DATASET_STEPS
    daily_starts: list[datetime] = [
        BASE_DATE + timedelta(hours=h)
        for h in range(0, DATASET_STEPS + DAILY_STEPS, DAILY_STEPS)
    ]
    n_days = len(daily_starts)

    records: list[dict] = []

    for base_ccy, quote_ccy in permutations(CURRENCIES, 2):
        pair = f"{base_ccy}/{quote_ccy}"
        if pair in ABSENT_PAIRS:
            logger.debug("Skipping absent FX pair: %s", pair)
            continue

        # Derive the USD-base rate for this pair
        if base_ccy == "USD" and quote_ccy in _USD_BASE:
            mean_rate, vol = _USD_BASE[quote_ccy]
        elif quote_ccy == "USD" and base_ccy in _USD_BASE:
            base_mean, base_vol = _USD_BASE[base_ccy]
            mean_rate = 1.0 / base_mean
            vol = base_vol
        elif base_ccy in _USD_BASE and quote_ccy in _USD_BASE:
            # Cross rate: base/USD * USD/quote
            bm, bv = _USD_BASE[base_ccy]
            qm, qv = _USD_BASE[quote_ccy]
            mean_rate = (1.0 / bm) * qm
            vol = bv + qv  # approximate combined volatility
        else:
            # USD/USD or identical pair (permutations won't produce this)
            mean_rate = 1.0
            vol = 0.001

        # Geometric random walk
        log_returns = rng.normal(0, vol, size=n_days)
        rates_float = mean_rate * np.exp(np.cumsum(log_returns))

        for ts, rate_f in zip(daily_starts, rates_float):
            records.append(
                {
                    "currency_pair": pair,
                    "rate": _rate_to_str(float(rate_f)),
                    "effective_timestamp": ts.isoformat(),
                }
            )

    logger.info("Generated %d FX rate rows for %d pairs", len(records), len(records) // n_days)
    return records


def write_fx_rates_parquet(records: list[dict], output_path: Path) -> None:
    """
    Write FX rate records to a Parquet file.

    Parameters
    ----------
    records : list[dict]
        Output of :func:`generate_fx_rates`.
    output_path : Path
        Destination .parquet file path.
    """
    schema = pa.schema(
        [
            pa.field("currency_pair", pa.string()),
            pa.field("rate", pa.string()),
            pa.field("effective_timestamp", pa.string()),
        ]
    )
    df = pd.DataFrame(records)
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    logger.info("FX rates written to %s (%d rows)", output_path, len(records))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from src.config import settings

    out = settings.processed_dir / "fx_rates.parquet"
    rates = generate_fx_rates(seed=settings.random_seed)
    write_fx_rates_parquet(rates, out)
    print(f"FX rates generated: {len(rates)} rows")
    print("Sample:", rates[:2])
