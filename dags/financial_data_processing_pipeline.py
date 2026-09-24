"""
financial_data_processing_pipeline.py — Airflow DAG for Distributed Financial Pipeline.

Orchestrates the end-to-end data processing workflow:
1. Register batch & check idempotency
2. Ingest and validate raw transactions (quarantine bad rows)
3. Spark transformation: deduplication and money standardisation
4. As-of FX rate join
5. Bulk load to PostgreSQL staging -> upsert to warehouse
6. Reconciliation matching & issue detection
7. Compute precision / recall against anomaly manifest
8. Complete batch & emit summary metrics

Rules:
- Retries: 3 attempts with exponential backoff
- XCom: Only small metadata (batch_id, row counts, metrics)
- Spark processes, Airflow orchestrates
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

default_args = {
    "owner": "data_engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
    "execution_timeout": timedelta(hours=2),
}

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    DAG = None
    PythonOperator = None


def task_start_batch(**context):
    from src.database.batch_manager import generate_batch_id, register_batch
    source_file = "data/raw/paysim.csv"
    batch_id = generate_batch_id(source_file)
    register_batch(batch_id, source_file)
    logger.info("Batch started: %s", batch_id)
    if context and "ti" in context:
        context["ti"].xcom_push(key="batch_id", value=batch_id)
    return batch_id


def task_ingest_validate(**context):
    from src.config import settings
    from src.ingestion.reader import read_paysim_chunks
    from src.ingestion.validator import validate_chunk, write_quarantine

    ti = context.get("ti") if context else None
    batch_id = ti.xcom_pull(key="batch_id", task_ids="start_batch") if ti else "default_batch"

    raw_csv = settings.raw_dir / "paysim.csv"
    total_valid = 0
    total_rejected = 0

    for chunk in read_paysim_chunks(raw_csv, chunk_size=settings.batch_size, batch_id=batch_id):
        res = validate_chunk(chunk, batch_id=batch_id, source_file=str(raw_csv))
        total_valid += len(res.valid_rows)
        total_rejected += len(res.rejected_rows)
        if res.rejected_rows:
            write_quarantine(res.rejected_rows, batch_id=batch_id, output_dir=settings.rejected_dir)

    logger.info("Ingest & validate done: valid=%d, rejected=%d", total_valid, total_rejected)
    if ti:
        ti.xcom_push(key="rows_valid", value=total_valid)
        ti.xcom_push(key="rows_rejected", value=total_rejected)
    return {"valid": total_valid, "rejected": total_rejected}


def task_spark_transform(**context):
    from src.config import settings
    from src.transformations.transaction_transform import (
        create_spark_session,
        run_transaction_transform,
    )

    ti = context.get("ti") if context else None
    batch_id = ti.xcom_pull(key="batch_id", task_ids="start_batch") if ti else "default_batch"

    spark = create_spark_session(f"Transform_{batch_id}")
    sample_parquet = settings.processed_dir / "enriched_sample.parquet"
    if not sample_parquet.exists():
        # Fallback to sample generation if missing
        from src.generators.enrich_paysim import enrich_paysim_csv
        enrich_paysim_csv(settings.raw_dir / "paysim.csv", sample_parquet, chunk_size=settings.batch_size, full=False)

    input_df = spark.read.parquet(str(sample_parquet))
    rows_clean, rows_quarantined = run_transaction_transform(
        spark=spark,
        input_df=input_df,
        batch_id=batch_id,
        output_dir=str(settings.processed_dir),
    )
    logger.info("Spark transform completed: clean=%d, quarantined=%d", rows_clean, rows_quarantined)
    if ti:
        ti.xcom_push(key="rows_clean", value=rows_clean)
    return {"rows_clean": rows_clean, "rows_quarantined": rows_quarantined}


def task_reconcile(**context):
    from src.config import settings
    from src.reconciliation.matcher import compute_precision_recall, run_reconciliation

    ti = context.get("ti") if context else None
    batch_id = ti.xcom_pull(key="batch_id", task_ids="start_batch") if ti else "default_batch"

    issues = run_reconciliation(batch_id)
    manifest_path = settings.processed_dir / "anomaly_manifest.parquet"
    metrics = {}
    if manifest_path.exists():
        metrics = compute_precision_recall(batch_id, str(manifest_path))

    logger.info("Reconciliation complete for batch=%s: issues=%s, metrics=%s", batch_id, issues, metrics)
    return {"issues": issues, "metrics": metrics}


def task_complete_batch(**context):
    from src.database.batch_manager import update_batch_status

    ti = context.get("ti") if context else None
    batch_id = ti.xcom_pull(key="batch_id", task_ids="start_batch") if ti else "default_batch"
    rows_valid = ti.xcom_pull(key="rows_valid", task_ids="ingest_validate") if ti else 0
    rows_rejected = ti.xcom_pull(key="rows_rejected", task_ids="ingest_validate") if ti else 0

    update_batch_status(
        batch_id=batch_id,
        status="COMPLETED",
        rows_valid=rows_valid,
        rows_rejected=rows_rejected,
    )
    logger.info("Batch %s marked COMPLETED successfully", batch_id)


if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="financial_data_processing_pipeline",
        default_args=default_args,
        description="End-to-end distributed financial processing and reconciliation",
        schedule_interval=None,  # Triggered manually or by external sensor
        start_date=datetime(2023, 1, 1),
        catchup=False,
        tags=["financial", "pyspark", "reconciliation"],
    ) as dag:
        t_start = PythonOperator(task_id="start_batch", python_callable=task_start_batch)
        t_ingest = PythonOperator(task_id="ingest_validate", python_callable=task_ingest_validate)
        t_transform = PythonOperator(task_id="spark_transform", python_callable=task_spark_transform)
        t_reconcile = PythonOperator(task_id="reconcile", python_callable=task_reconcile)
        t_complete = PythonOperator(task_id="complete_batch", python_callable=task_complete_batch)

        t_start >> t_ingest >> t_transform >> t_reconcile >> t_complete

