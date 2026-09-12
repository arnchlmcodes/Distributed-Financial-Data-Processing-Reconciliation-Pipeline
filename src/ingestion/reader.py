"""
reader.py — Read CSV and Parquet data sources with schema enforcement.
CSV is read in chunks to avoid full 6.3M row memory load.
Parquet is read with PySpark using explicit schema.
"""
from __future__ import annotations

import csv
import logging
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

# Expected column order in the PaySim CSV header
PAYSIM_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


def read_paysim_chunks(
    csv_path: Path,
    chunk_size: int = 50_000,
    batch_id: str = "batch_001",
) -> Iterator[list[dict]]:
    """Read PaySim CSV in chunks. Yields lists of raw dicts.

    The CSV is streamed row-by-row using the stdlib ``csv`` module so that
    the full 6.3 M-row file is never loaded into memory at once.

    Args:
        csv_path: Path to the PaySim CSV file.
        chunk_size: Number of rows per yielded chunk.
        batch_id: Identifier for the current ingestion batch (included so
            callers can tag rows immediately).

    Yields:
        Lists of dicts where keys are ``PAYSIM_COLUMNS`` plus ``batch_id``.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"PaySim CSV not found: {csv_path}")

    logger.info("Opening PaySim CSV: %s (chunk_size=%d)", csv_path, chunk_size)

    chunk: list[dict] = []
    total_rows = 0

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)

        # Validate header columns are present
        if reader.fieldnames is None:
            raise ValueError(f"CSV file appears to be empty: {csv_path}")

        missing = set(PAYSIM_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        for row in reader:
            # Keep only the columns we care about; attach batch_id
            record = {col: row.get(col) for col in PAYSIM_COLUMNS}
            record["batch_id"] = batch_id
            chunk.append(record)
            total_rows += 1

            if len(chunk) >= chunk_size:
                logger.debug("Yielding chunk of %d rows (total so far: %d)", len(chunk), total_rows)
                yield chunk
                chunk = []

    # Yield remaining rows
    if chunk:
        logger.debug("Yielding final chunk of %d rows (total: %d)", len(chunk), total_rows)
        yield chunk

    logger.info("Finished reading PaySim CSV: %d total rows", total_rows)


def read_parquet_with_schema(
    parquet_path: Path,
    spark,  # SparkSession
    schema,  # PySpark StructType
):
    """Read a Parquet file with explicit PySpark schema enforcement.

    ``inferSchema`` is never used. The caller-supplied ``schema`` is applied
    directly so that column types are guaranteed without scanning the file.

    Args:
        parquet_path: Path to the Parquet file (or directory).
        spark: Active ``SparkSession``.
        schema: PySpark ``StructType`` describing the expected columns.

    Returns:
        A PySpark ``DataFrame`` with the given schema.

    Raises:
        FileNotFoundError: If ``parquet_path`` does not exist.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet path not found: {parquet_path}")

    logger.info("Reading parquet with explicit schema: %s", parquet_path)
    df = spark.read.schema(schema).parquet(str(parquet_path))
    logger.info("Parquet read complete. Schema enforced: %s", schema.simpleString())
    return df
