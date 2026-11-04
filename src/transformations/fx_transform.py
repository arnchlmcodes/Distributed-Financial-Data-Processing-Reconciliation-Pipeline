"""
fx_transform.py — As-of FX rate join for transaction amounts.

Rules:
- FX join: latest rate with effective_timestamp <= transaction_timestamp (never join on pair alone)
- Use Spark window function (row_number) for as-of join
- Broadcast the FX rates table (small, fits in memory)
- Flag MISSING_FX_RATE when no rate found for a pair
- Rates stored at DecimalType(18,6); converted amounts at DecimalType(18,2)
- Absent pairs (NGN/KES, KES/NGN) will always produce MISSING_FX_RATE
"""
from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType, StringType
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)

MISSING_FX_SENTINEL = "MISSING_FX_RATE"


def load_fx_rates(spark: SparkSession, fx_parquet_path: str) -> DataFrame:
    """
    Load FX rates from Parquet with explicit schema.
    Returns a small DataFrame suitable for broadcasting.
    """

    from src.ingestion.schema import FX_RATE_SCHEMA

    df = (
        spark.read.schema(FX_RATE_SCHEMA)
        .parquet(fx_parquet_path)
    )
    row_count = df.count()
    logger.info("Loaded %d FX rate records from %s", row_count, fx_parquet_path)
    return df


def apply_fx_rates(
    txn_df: DataFrame,
    fx_df: DataFrame,
) -> DataFrame:
    """
    Perform as-of FX join: for each transaction, find the latest FX rate
    where rate.effective_timestamp <= transaction.transaction_timestamp
    AND rate.currency_pair == transaction.fx_pair.

    Strategy:
    1. Cross-join transactions with FX rates on currency_pair (broadcast FX)
    2. Filter: rate.effective_timestamp <= txn.transaction_timestamp
    3. Window: row_number() over (partition by txn_id ORDER BY rate.effective_timestamp DESC)
    4. Keep row_number == 1 → closest-before rate
    5. Left-join back to get MISSING_FX_RATE for unmatched pairs

    Args:
        txn_df: Transactions with columns: transaction_id, fx_pair, transaction_timestamp, amount
        fx_df: FX rates with columns: currency_pair, rate, effective_timestamp

    Returns:
        txn_df enriched with: fx_rate (Decimal 18,6), amount_usd (Decimal 18,2), fx_issue (str|null)
    """
    # Broadcast the FX table (typically small: ~30 pairs × ~30 days = ~900 rows)
    fx_broadcast = F.broadcast(fx_df)

    # Rename FX columns to avoid ambiguity
    fx_renamed = fx_broadcast.select(
        F.col("currency_pair").alias("_fx_pair"),
        F.col("rate").alias("_fx_rate"),
        F.col("effective_timestamp").alias("_fx_effective_ts"),
    )

    # Join on pair + as-of condition
    joined = txn_df.join(
        fx_renamed,
        on=(
            (F.col("fx_pair") == F.col("_fx_pair")) &
            (F.col("_fx_effective_ts") <= F.col("transaction_timestamp"))
        ),
        how="left",
    )

    # Window: pick the most recent rate per transaction
    w_aof = Window.partitionBy("transaction_id").orderBy(F.col("_fx_effective_ts").desc())
    joined = joined.withColumn("_rn", F.row_number().over(w_aof))

    # Keep only the best (most recent valid) rate
    best_rate = joined.filter(
        (F.col("_rn") == 1) | F.col("_fx_rate").isNull()
    ).drop("_rn")

    # Compute amount in USD: rate is defined as units of USD per 1 unit of source currency
    # Formula: amount_usd = (amount * fx_rate).cast(DecimalType(18, 2))
    # If rate is null (missing pair or pre-dates first available rate) → flag MISSING_FX_RATE
    best_rate = best_rate.withColumn(
        "fx_rate",
        F.col("_fx_rate").cast(DecimalType(18, 6)),
    ).withColumn(
        "amount_usd",
        F.when(
            F.col("_fx_rate").isNotNull(),
            (F.col("amount") * F.col("_fx_rate")).cast(DecimalType(18, 2)),
        ).otherwise(F.lit(None).cast(DecimalType(18, 2))),
    ).withColumn(
        "fx_issue",
        F.when(F.col("_fx_rate").isNull(), F.lit(MISSING_FX_SENTINEL))
        .otherwise(F.lit(None).cast(StringType())),
    ).drop("_fx_pair", "_fx_rate", "_fx_effective_ts")

    missing_count = best_rate.filter(F.col("fx_issue") == MISSING_FX_SENTINEL).count()
    total = best_rate.count()
    logger.info(
        "FX join complete — total: %d, missing FX rate: %d (%.2f%%)",
        total,
        missing_count,
        100.0 * missing_count / total if total > 0 else 0.0,
    )

    return best_rate

