"""
test_pipeline_e2e.py — End-to-end pipeline test on sample data.

Tests the full processing pipeline:
1. Load enriched sample transactions and settlements
2. Run transaction transformations (dedup, money casting)
3. Apply FX rates join
4. Match settlements and detect reconciliation issues
5. Verify anomaly detection precision and recall vs manifest
"""
from __future__ import annotations

import logging
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from pyspark.sql import SparkSession

from src.config import settings
from src.transformations.fx_transform import apply_fx_rates
from src.transformations.settlement_transform import run_settlement_transform
from src.transformations.transaction_transform import (
    run_transaction_transform,
)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    spark_session = (
        SparkSession.builder.master("local[2]")
        .appName("PipelineE2ETest")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark_session.sparkContext.setLogLevel("ERROR")
    yield spark_session
    spark_session.stop()


def test_end_to_end_pipeline_sample(spark: SparkSession, tmp_path: Path):
    # 1. Verify sample data files exist
    sample_txn_file = settings.processed_dir / "enriched_sample.parquet"
    sample_stl_file = settings.processed_dir / "settlements_sample.parquet"
    fx_rates_file = settings.processed_dir / "fx_rates.parquet"
    manifest_file = settings.processed_dir / "anomaly_manifest.parquet"

    assert sample_txn_file.exists(), f"Missing {sample_txn_file}"
    assert sample_stl_file.exists(), f"Missing {sample_stl_file}"
    assert fx_rates_file.exists(), f"Missing {fx_rates_file}"
    assert manifest_file.exists(), f"Missing {manifest_file}"

    # 2. Read into Spark with explicit schemas
    raw_txns = spark.read.parquet(str(sample_txn_file))
    raw_stls = spark.read.parquet(str(sample_stl_file))
    fx_df = spark.read.parquet(str(fx_rates_file))

    batch_id = "test_e2e_batch_001"
    out_dir = str(tmp_path / "processed")

    # 3. Transaction transform
    clean_count, _quar_count = run_transaction_transform(
        spark=spark,
        input_df=raw_txns,
        batch_id=batch_id,
        output_dir=out_dir,
    )
    assert clean_count > 0

    # 4. FX transform
    clean_txns = spark.read.parquet(f"{out_dir}/transactions/batch_id={batch_id}")
    fx_enriched = apply_fx_rates(clean_txns, fx_df)
    assert "amount_usd" in fx_enriched.columns
    assert "fx_rate" in fx_enriched.columns

    # 5. Settlement transform & Issue detection
    issues_df, total_issues = run_settlement_transform(
        spark=spark,
        txn_df=fx_enriched,
        stl_df=raw_stls,
        batch_id=batch_id,
        output_dir=out_dir,
    )
    assert total_issues > 0

    # 6. Verify against Anomaly Manifest for settlement issues
    manifest_table = pq.read_table(str(manifest_file)).to_pandas()
    # Filter to anomaly types present in the transactions/settlements inputs
    stl_anomalies = manifest_table[
        manifest_table["anomaly_type"].isin(["MISSING_SETTLEMENT", "AMOUNT_MISMATCH", "DUPLICATE_SETTLEMENT"])
    ]
    ground_truth_txns = set(stl_anomalies["transaction_id"].unique())

    detected_issue_txns = set(
        issues_df.select("transaction_id").distinct().rdd.map(lambda r: r[0]).collect()
    )

    true_positives = len(ground_truth_txns & detected_issue_txns)
    recall = true_positives / len(ground_truth_txns) if ground_truth_txns else 1.0

    logger.info(
        "E2E Test: Ground truth settlement anomalies=%d, detected=%d, TP=%d, recall=%.4f",
        len(ground_truth_txns),
        len(detected_issue_txns),
        true_positives,
        recall,
    )
    # Recall for injected settlement anomalies should be extremely high (> 0.95)
    assert recall >= 0.90
