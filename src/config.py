"""
config.py — Central configuration loader.
Reads config/settings.yaml and .env, exposes a single Settings object.
"""
from __future__ import annotations

import logging
import os
import sys
from decimal import Decimal
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Ensure Spark workers always find the current Python virtualenv
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
if os.path.exists(r"C:\hadoop\bin\winutils.exe"):
    os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
    os.environ.setdefault("hadoop.home.dir", r"C:\hadoop")
    if r"C:\hadoop\bin" not in os.environ.get("PATH", ""):
        os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ.get("PATH", "")

logger = logging.getLogger(__name__)

# Load .env once at import time
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH, override=False)

_SETTINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


def _load_yaml() -> dict:
    with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Settings:
    """Typed access to config/settings.yaml + .env overrides."""

    def __init__(self) -> None:
        raw = _load_yaml()
        proj = raw.get("project", {})
        data = raw.get("data", {})
        proc = raw.get("processing", {})
        anom = raw.get("anomalies", {})

        self.project_name: str = proj.get("name", "distributed-financial-pipeline")
        self.random_seed: int = int(os.getenv("RANDOM_SEED", proj.get("random_seed", 42)))

        self.raw_dir: Path = Path(data.get("raw_dir", "data/raw"))
        self.processed_dir: Path = Path(data.get("processed_dir", "data/processed"))
        self.rejected_dir: Path = Path(data.get("rejected_dir", "data/rejected"))

        self.batch_size: int = int(os.getenv("BATCH_SIZE", proc.get("batch_size", 50_000)))
        self.money_precision: int = int(proc.get("money_precision", 2))
        self.reconciliation_tolerance: Decimal = Decimal(
            str(os.getenv("RECONCILIATION_TOLERANCE", proc.get("reconciliation_tolerance", "0.01")))
        )

        # Anomaly injection rates
        self.missing_settlement_rate: float = float(anom.get("missing_settlement_rate", 0.02))
        self.duplicate_transaction_rate: float = float(anom.get("duplicate_transaction_rate", 0.01))
        self.duplicate_settlement_rate: float = float(anom.get("duplicate_settlement_rate", 0.01))
        self.amount_mismatch_rate: float = float(anom.get("amount_mismatch_rate", 0.01))
        self.invalid_record_rate: float = float(anom.get("invalid_record_rate", 0.005))

        # Database (from .env)
        self.pg_db: str = os.getenv("POSTGRES_DB", "financial_pipeline")
        self.pg_user: str = os.getenv("POSTGRES_USER", "pipeline_user")
        self.pg_password: str = os.getenv("POSTGRES_PASSWORD", "change_me")
        self.pg_host: str = os.getenv("POSTGRES_HOST", "localhost")
        self.pg_port: int = int(os.getenv("POSTGRES_PORT", "5432"))

        # Spark
        self.spark_master: str = os.getenv("SPARK_MASTER", "local[*]")
        self.spark_shuffle_partitions: int = 16
        self.spark_driver_memory: str = "4g"

        # Timestamps
        # PaySim step → timestamp: base_date + step * 1 hour
        self.base_date: str = raw.get("data", {}).get("base_date", "2023-01-01T00:00:00")

        # Currencies supported (some pairs intentionally absent to create MISSING_FX_RATE)
        self.currencies: list[str] = ["USD", "EUR", "GBP", "INR", "NGN", "KES"]
        self.absent_fx_pairs: list[str] = ["NGN/KES", "KES/NGN"]

        logger.info("Settings loaded from %s", _SETTINGS_PATH)

    @property
    def pg_dsn(self) -> str:
        """psycopg3-compatible DSN string."""
        return (
            f"host={self.pg_host} port={self.pg_port} "
            f"dbname={self.pg_db} user={self.pg_user} password={self.pg_password}"
        )


# Module-level singleton
settings = Settings()

