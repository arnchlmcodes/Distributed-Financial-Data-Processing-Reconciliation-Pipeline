# Pipeline Execution Script: runs the end-to-end processing pipeline locally
"""
pipeline_runner.py — Standalone CLI orchestrator for `make pipeline`.
Runs:
1. Ingestion & Validation -> Quarantine
2. PySpark Deduplication & Transformations
3. As-of FX Join
4. Settlement Matching & Issue Detection
5. Metrics calculation against Anomaly Manifest
"""
import logging
import os
import sys
import time
from datetime import UTC, datetime

# Set environment
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
if os.path.exists(r"C:\hadoop\bin\winutils.exe"):
    os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
    os.environ.setdefault("hadoop.home.dir", r"C:\hadoop")

import pyarrow.parquet as pq

from src.config import settings
from src.transformations.fx_transform import apply_fx_rates
from src.transformations.settlement_transform import run_settlement_transform
from src.transformations.transaction_transform import (
    create_spark_session,
    run_transaction_transform,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("pipeline")


def run_pipeline(batch_id: str | None = None):
    if not batch_id:
        batch_id = f"batch_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}"

    logger.info("==================================================")
    logger.info("STARTING FINANCIAL PIPELINE RUN: %s", batch_id)
    logger.info("==================================================")

    sample_txn_file = settings.processed_dir / "enriched_sample.parquet"
    sample_stl_file = settings.processed_dir / "settlements_sample.parquet"
    fx_rates_file = settings.processed_dir / "fx_rates.parquet"
    manifest_file = settings.processed_dir / "anomaly_manifest.parquet"

    assert sample_txn_file.exists(), f"Missing {sample_txn_file}"
    assert sample_stl_file.exists(), f"Missing {sample_stl_file}"
    assert fx_rates_file.exists(), f"Missing {fx_rates_file}"

    spark = create_spark_session("FinancialPipelineLocal")

    start_t = time.perf_counter()

    # Step 1: Read raw sources
    raw_txns = spark.read.parquet(str(sample_txn_file))
    raw_stls = spark.read.parquet(str(sample_stl_file))
    fx_df = spark.read.parquet(str(fx_rates_file))

    out_dir = str(settings.processed_dir / "pipeline_output")

    # Step 2: Transaction transformations & dedup
    logger.info("Executing Transaction Standardisation and Deduplication...")
    clean_count, quar_count = run_transaction_transform(
        spark=spark,
        input_df=raw_txns,
        batch_id=batch_id,
        output_dir=out_dir,
    )
    logger.info("Transactions processed: %d valid, %d quarantined", clean_count, quar_count)

    # Step 3: As-of FX Join
    logger.info("Executing As-of FX Join (Windowed Broadcast)...")
    clean_txns = spark.read.parquet(f"{out_dir}/transactions/batch_id={batch_id}")
    fx_enriched = apply_fx_rates(clean_txns, fx_df)

    # Step 4: Settlement matching & issue detection
    logger.info("Executing Settlement Matching and Reconciliation...")
    issues_df, total_issues = run_settlement_transform(
        spark=spark,
        txn_df=fx_enriched,
        stl_df=raw_stls,
        batch_id=batch_id,
        output_dir=out_dir,
    )

    elapsed = round(time.perf_counter() - start_t, 2)

    # Step 5: Recall / Precision vs Manifest
    manifest_table = pq.read_table(str(manifest_file)).to_pandas()
    stl_anomalies = manifest_table[
        manifest_table["anomaly_type"].isin(["MISSING_SETTLEMENT", "AMOUNT_MISMATCH", "DUPLICATE_SETTLEMENT"])
    ]
    ground_truth_txns = set(stl_anomalies["transaction_id"].unique())
    detected_issue_txns = set(
        issues_df.select("transaction_id").distinct().rdd.map(lambda r: r[0]).collect()
    )
    tp = len(ground_truth_txns & detected_issue_txns)
    recall = tp / len(ground_truth_txns) if ground_truth_txns else 1.0

    logger.info("==================================================")
    logger.info("PIPELINE COMPLETED SUCCESSFULLY IN %.2f SECONDS", elapsed)
    logger.info("Transactions Clean: %d | Quarantined: %d", clean_count, quar_count)
    logger.info("Reconciliation Issues Flagged: %d", total_issues)
    logger.info("Anomaly Detection Recall: %.2f%% (%d / %d ground-truth settlement anomalies)", recall * 100, tp, len(ground_truth_txns))
    logger.info("==================================================")


if __name__ == "__main__":
    run_pipeline()
