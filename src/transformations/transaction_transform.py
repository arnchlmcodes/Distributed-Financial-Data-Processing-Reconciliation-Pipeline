"""
transaction_transform.py — PySpark transaction deduplication and standardisation.

Rules:
- Exact duplicates (same transaction_id, same content) → keep earliest by ingestion order
- Conflicting duplicates (same transaction_id, different content) → quarantine both
- All money columns cast to DecimalType(18,2) — never float
- shuffle_partitions tuned via config
"""
from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType, StringType, TimestampType

from src.config import settings

logger = logging.getLogger(__name__)

MONEY_COLUMNS = [
    "amount", "old_balance_orig", "new_balance_orig",
    "old_balance_dest", "new_balance_dest",
]


def create_spark_session(app_name: str = "FinancialPipeline") -> SparkSession:
    """Create a configured SparkSession. Uses local[*] by default."""
    spark = (
        SparkSession.builder.appName(app_name)
        .master(settings.spark_master)
        .config("spark.sql.shuffle.partitions", str(settings.spark_shuffle_partitions))
        .config("spark.driver.memory", settings.spark_driver_memory)
        .config("spark.sql.decimal.handleLegacyBigDecimal", "true")
        # Disable automatic schema evolution warnings
        .config("spark.sql.parquet.mergeSchema", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info("SparkSession created: master=%s", settings.spark_master)
    return spark


def cast_money_columns(df: DataFrame) -> DataFrame:
    """
    Cast all money columns to DecimalType(18,2) using ROUND_HALF_UP.
    Input columns may be StringType (from parquet) or already Decimal.
    """
    for col_name in MONEY_COLUMNS:
        if col_name in df.columns:
            df = df.withColumn(
                col_name,
                F.col(col_name).cast(DecimalType(18, 2)),
            )
    logger.debug("Money columns cast to DecimalType(18,2): %s", MONEY_COLUMNS)
    return df


def deduplicate_transactions(
    df: DataFrame,
    batch_id: str,
    spark: SparkSession,
) -> tuple[DataFrame, DataFrame]:
    """
    Deduplicate transaction records.

    Returns:
        (clean_df, quarantine_df)
        - clean_df: deduplicated valid transactions
        - quarantine_df: conflicting duplicates with reason column
    """
    # Add row number for deterministic ordering (earliest ingestion wins)
    w_ingestion = Window.partitionBy("transaction_id").orderBy(
        F.monotonically_increasing_id()
    )
    df = df.withColumn("_row_num", F.row_number().over(w_ingestion))

    # Compute a content hash for each row (excluding row_num itself)
    content_cols = [c for c in df.columns if not c.startswith("_")]
    df = df.withColumn(
        "_content_hash",
        F.md5(F.concat_ws("|", *[F.coalesce(F.col(c).cast(StringType()), F.lit("NULL")) for c in content_cols]))
    )

    # Count distinct content hashes per transaction_id
    hash_counts = (
        df.groupBy("transaction_id")
        .agg(
            F.countDistinct("_content_hash").alias("_n_distinct_hashes"),
            F.count("*").alias("_total_count"),
        )
    )

    df = df.join(hash_counts, on="transaction_id", how="left")

    # Conflicting duplicates: same id, different content → quarantine ALL copies
    conflicting = df.filter(F.col("_n_distinct_hashes") > 1)
    quarantine_df = conflicting.withColumn(
        "quarantine_reason", F.lit("CONFLICTING_DUPLICATE")
    ).withColumn(
        "batch_id_quarantine", F.lit(batch_id)
    )

    # Exact duplicates: same id, same content → keep first by row_num
    non_conflicting = df.filter(F.col("_n_distinct_hashes") == 1)
    clean_df = non_conflicting.filter(F.col("_row_num") == 1)

    # Drop helper columns
    drop_cols = ["_row_num", "_content_hash", "_n_distinct_hashes", "_total_count"]
    clean_df = clean_df.drop(*drop_cols)
    quarantine_df = quarantine_df.drop(*drop_cols)

    exact_dup_count = non_conflicting.filter(F.col("_row_num") > 1).count()
    conflict_count = conflicting.select("transaction_id").distinct().count()

    logger.info(
        "Dedup complete — clean: %d, exact dups dropped: %d, conflicting quarantined: %d",
        clean_df.count(),
        exact_dup_count,
        conflict_count,
    )

    return clean_df, quarantine_df


def standardise_timestamps(df: DataFrame, base_date: str) -> DataFrame:
    """
    Convert PaySim `step` column to `transaction_timestamp`.
    Formula: base_date + step * 1 hour (UTC).
    """
    base_ts = F.to_timestamp(F.lit(base_date))
    df = df.withColumn(
        "transaction_timestamp",
        (base_ts.cast("long") + F.col("step").cast("long") * 3600).cast(TimestampType()),
    )
    return df


def run_transaction_transform(
    spark: SparkSession,
    input_df: DataFrame,
    batch_id: str,
    output_dir: str,
) -> tuple[int, int]:
    """
    Full transaction transformation pipeline.

    Args:
        spark: SparkSession
        input_df: Raw transaction DataFrame (schema already applied)
        batch_id: Pipeline batch identifier
        output_dir: Root dir for Parquet output

    Returns:
        (rows_clean, rows_quarantined)
    """
    logger.info("Starting transaction transform for batch=%s", batch_id)

    # 1. Cast money columns
    df = cast_money_columns(input_df)

    # 2. Standardise timestamps if not already present
    if "transaction_timestamp" not in df.columns:
        df = standardise_timestamps(df, settings.base_date)

    # 3. Add batch_id if not present
    if "batch_id" not in df.columns:
        df = df.withColumn("batch_id", F.lit(batch_id))

    # 4. Deduplicate
    clean_df, quarantine_df = deduplicate_transactions(df, batch_id, spark)

    rows_clean = clean_df.count()
    rows_quarantined = quarantine_df.count()

    # 5. Write clean Parquet partitioned by batch_id
    clean_path = f"{output_dir}/transactions/batch_id={batch_id}"
    (
        clean_df.write.mode("overwrite")
        .option("compression", "snappy")
        .parquet(clean_path)
    )
    logger.info("Clean transactions written to %s (%d rows)", clean_path, rows_clean)

    # 6. Write quarantine Parquet
    if rows_quarantined > 0:
        quar_path = f"{output_dir}/quarantine/transactions/batch_id={batch_id}"
        (
            quarantine_df.write.mode("overwrite")
            .option("compression", "snappy")
            .parquet(quar_path)
        )
        logger.warning("Quarantined %d conflicting transactions → %s", rows_quarantined, quar_path)

    return rows_clean, rows_quarantined

