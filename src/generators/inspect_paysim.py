"""
inspect_paysim.py — Dataset inspection and profiling for the PaySim CSV.

Reads data/raw/paysim.csv in configurable chunks and writes a JSON
profiling report to data/processed/profile.json.
Money statistics (amount) are computed using decimal.Decimal; floats
are never used for monetary values.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow running as  `python src/generators/inspect_paysim.py`  from the repo
# root without installing the package.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Columns that form a "logical duplicate" key
_DUPE_KEY_COLS = ["step", "type", "amount", "nameOrig", "nameDest"]


def _decimal(value: str) -> Decimal:
    """Parse a CSV string value into a Decimal, stripping whitespace."""
    return Decimal(value.strip())


def profile_dataset(csv_path: Path, chunk_size: int = 50_000) -> dict:
    """Profile the PaySim CSV in chunks and return a summary dictionary.

    Parameters
    ----------
    csv_path:
        Absolute or relative path to the CSV file.
    chunk_size:
        Number of rows per chunk. Defaults to 50 000.

    Returns
    -------
    dict with keys:
        row_count         – total rows processed
        column_types      – {column: inferred_dtype_string}
        null_counts       – {column: int}
        type_distribution – {transaction_type: int}
        fraud_distribution – {0: int, 1: int} (isFraud)
        amount_stats      – {min, max, mean} as Decimal strings
        duplicate_count   – int, duplicate key combos
    """
    import csv  # stdlib — lightweight; avoids pandas/numpy for this step

    logger.info("Starting profile of %s (chunk_size=%d)", csv_path, chunk_size)

    row_count: int = 0
    null_counts: dict[str, int] = defaultdict(int)
    type_distribution: dict[str, int] = defaultdict(int)
    fraud_distribution: dict[str, int] = defaultdict(int)

    # Amount accumulators — all Decimal
    amount_min: Decimal | None = None
    amount_max: Decimal | None = None
    amount_sum: Decimal = Decimal(0)

    # Duplicate detection: store seen keys in a set.
    # Each key is a tuple; the set fits in memory (≤6.3 M entries ≈ ~500 MB
    # worst case, acceptable for a one-time profiling run).
    seen_keys: set[tuple] = set()
    dupe_count: int = 0

    columns: list[str] = []
    chunk: list[dict] = []

    def _process_chunk(rows: list[dict]) -> None:
        """Update accumulators from a batch of rows."""
        nonlocal amount_min, amount_max, amount_sum, dupe_count

        for row in rows:
            # Null counts
            for col, val in row.items():
                if val == "" or val is None:
                    null_counts[col] += 1

            # type distribution
            type_distribution[row.get("type", "")] += 1

            # fraud distribution
            fraud_val = row.get("isFraud", "").strip()
            if fraud_val in ("0", "1"):
                fraud_distribution[int(fraud_val)] += 1

            # amount — Decimal only
            raw_amount = row.get("amount", "").strip()
            if raw_amount:
                try:
                    amt = _decimal(raw_amount)
                    amount_sum += amt
                    if amount_min is None or amt < amount_min:
                        amount_min = amt
                    if amount_max is None or amt > amount_max:
                        amount_max = amt
                except (InvalidOperation, ValueError, TypeError):
                    null_counts["amount"] += 1  # treat unparseable as null

            # Duplicate key
            key = tuple(row.get(c, "").strip() for c in _DUPE_KEY_COLS)
            if key in seen_keys:
                dupe_count += 1
            else:
                seen_keys.add(key)

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        columns = list(reader.fieldnames or [])
        logger.info("Columns detected: %s", columns)

        for row in reader:
            chunk.append(row)
            row_count += 1

            if len(chunk) >= chunk_size:
                _process_chunk(chunk)
                chunk = []
                logger.info("  … processed %d rows so far", row_count)

        # Tail chunk
        if chunk:
            _process_chunk(chunk)
            logger.info("  … processed %d rows (final chunk)", row_count)

    # Compute mean — guard against zero rows
    if row_count > 0 and amount_min is not None:
        amount_mean = (amount_sum / Decimal(row_count)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
    else:
        amount_mean = Decimal(0)

    # Column types: inferred from the CSV header names (all strings in a CSV,
    # but we label the known numeric / categorical ones for the report).
    _known_types: dict[str, str] = {
        "step": "int",
        "type": "str",
        "amount": "Decimal",
        "nameOrig": "str",
        "oldbalanceOrg": "Decimal",
        "newbalanceOrig": "Decimal",
        "nameDest": "str",
        "oldbalanceDest": "Decimal",
        "newbalanceDest": "Decimal",
        "isFraud": "int",
        "isFlaggedFraud": "int",
    }
    column_types = {col: _known_types.get(col, "str") for col in columns}

    result: dict = {
        "row_count": row_count,
        "column_types": column_types,
        "null_counts": dict(null_counts),
        "type_distribution": dict(type_distribution),
        "fraud_distribution": {str(k): v for k, v in fraud_distribution.items()},
        "amount_stats": {
            "min": str(amount_min) if amount_min is not None else "N/A",
            "max": str(amount_max) if amount_max is not None else "N/A",
            "mean": str(amount_mean),
        },
        "duplicate_count": dupe_count,
    }

    logger.info(
        "Profile complete: %d rows, %d duplicates, fraud=%s",
        row_count,
        dupe_count,
        dict(fraud_distribution),
    )
    return result


def _write_report(report: dict, out_path: Path) -> None:
    """Serialise *report* to JSON and write to *out_path*."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("Profile written to %s", out_path)


if __name__ == "__main__":
    csv_path = settings.raw_dir / "paysim.csv"
    out_path = settings.processed_dir / "profile.json"

    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)

    report = profile_dataset(csv_path, chunk_size=settings.batch_size)
    _write_report(report, out_path)

    # Quick summary to stdout via logger
    logger.info("=== SUMMARY ===")
    logger.info("Rows : %d", report["row_count"])
    logger.info("Nulls: %s", report["null_counts"])
    logger.info("Types: %s", report["type_distribution"])
    logger.info("Fraud: %s", report["fraud_distribution"])
    logger.info("Amt  : %s", report["amount_stats"])
    logger.info("Dupes: %d", report["duplicate_count"])
