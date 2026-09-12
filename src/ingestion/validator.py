"""
validator.py — Validate ingested records; write rejects to quarantine.

Rules:
- amount must be parseable as Decimal and >= 0
- required string fields must be non-null and non-empty
- type must be one of: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
- step must be integer >= 0
- is_fraud, is_flagged_fraud must be 0 or 1
- transaction_id must be non-null

Rejected rows go to data/rejected/ as parquet with full quarantine metadata.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

VALID_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}

# String fields that must be present and non-empty in a raw PaySim row
_REQUIRED_STRING_FIELDS = ("type", "nameOrig", "nameDest")

# Fields that represent money (read as strings from CSV, may be None/empty)
_MONEY_FIELDS = (
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)

# Quarantine PyArrow schema (mirrors QUARANTINE_SCHEMA in schema.py but uses
# pa.string() so we can write without a SparkSession)
_QUARANTINE_PA_SCHEMA = pa.schema([
    pa.field("batch_id", pa.string(), nullable=False),
    pa.field("source_file", pa.string(), nullable=False),
    pa.field("record_type", pa.string(), nullable=False),
    pa.field("payload", pa.string(), nullable=False),
    pa.field("reasons", pa.string(), nullable=False),   # JSON list
    pa.field("quarantined_at", pa.string(), nullable=False),  # ISO 8601
])


@dataclass
class ValidationResult:
    """Container returned by :func:`validate_chunk`."""

    valid_rows: list[dict] = field(default_factory=list)
    rejected_rows: list[dict] = field(default_factory=list)
    rejection_counts: dict[str, int] = field(default_factory=dict)


def _increment(counts: dict[str, int], reason: str) -> None:
    counts[reason] = counts.get(reason, 0) + 1


def _validate_row(row: dict[str, Any]) -> list[str]:
    """Return a list of validation failure reasons for a single raw row.

    An empty list means the row is valid.
    """
    reasons: list[str] = []

    # --- transaction_id ---
    txn_id = row.get("transaction_id")
    if txn_id is None or str(txn_id).strip() == "":
        reasons.append("missing transaction_id")

    # --- step ---
    step_raw = row.get("step")
    try:
        step_val = int(step_raw)  # type: ignore[arg-type]
        if step_val < 0:
            reasons.append("negative step")
    except (TypeError, ValueError):
        reasons.append("invalid step (not an integer)")

    # --- type ---
    txn_type = row.get("type")
    if not txn_type or str(txn_type).strip() == "":
        reasons.append("missing type")
    elif str(txn_type).strip() not in VALID_TYPES:
        reasons.append(f"invalid type: {txn_type!r}")

    # --- required string fields ---
    for fld in _REQUIRED_STRING_FIELDS:
        if fld == "type":
            continue  # already checked above
        val = row.get(fld)
        if val is None or str(val).strip() == "":
            reasons.append(f"missing required field: {fld}")

    # --- amount (mandatory, >= 0) ---
    amount_raw = row.get("amount")
    if amount_raw is None or str(amount_raw).strip() == "":
        reasons.append("missing amount")
    else:
        try:
            amount_val = Decimal(str(amount_raw).strip())
            if amount_val < Decimal(0):
                reasons.append("negative amount")
        except InvalidOperation:
            reasons.append(f"unparseable amount: {amount_raw!r}")

    # --- optional money fields (if present must be parseable) ---
    for mf in _MONEY_FIELDS:
        if mf == "amount":
            continue  # already handled
        val = row.get(mf)
        if val is not None and str(val).strip() != "":
            try:
                Decimal(str(val).strip())
            except InvalidOperation:
                reasons.append(f"unparseable money field {mf}: {val!r}")

    # --- is_fraud / is_flagged_fraud ---
    for flag_field in ("isFraud", "is_fraud"):
        if flag_field in row:
            flag_raw = row[flag_field]
            try:
                flag_val = int(flag_raw)
                if flag_val not in (0, 1):
                    reasons.append(f"invalid {flag_field}: {flag_raw!r} (must be 0 or 1)")
            except (TypeError, ValueError):
                reasons.append(f"unparseable {flag_field}: {flag_raw!r}")
            break  # only check whichever key is present

    for flag_field in ("isFlaggedFraud", "is_flagged_fraud"):
        if flag_field in row:
            flag_raw = row[flag_field]
            try:
                flag_val = int(flag_raw)
                if flag_val not in (0, 1):
                    reasons.append(f"invalid {flag_field}: {flag_raw!r} (must be 0 or 1)")
            except (TypeError, ValueError):
                reasons.append(f"unparseable {flag_field}: {flag_raw!r}")
            break

    return reasons


def validate_chunk(
    rows: list[dict],
    batch_id: str,
    source_file: str,
) -> ValidationResult:
    """Validate a chunk of raw transaction rows.

    Each row is checked against the business rules defined in this module.
    Valid rows are collected in ``ValidationResult.valid_rows``; failing rows
    are collected in ``ValidationResult.rejected_rows`` with quarantine
    metadata attached.

    Args:
        rows: List of raw transaction dicts (e.g. from :func:`read_paysim_chunks`).
        batch_id: Identifier for the current ingestion batch.
        source_file: Name/path of the originating file, stored in quarantine.

    Returns:
        A :class:`ValidationResult` with valid and rejected rows separated.
    """
    result = ValidationResult()
    now_iso = datetime.now(UTC).isoformat()

    for row in rows:
        reasons = _validate_row(row)

        if not reasons:
            result.valid_rows.append(row)
        else:
            # Tag rejected row with quarantine metadata
            quarantine_record = {
                "batch_id": batch_id,
                "source_file": source_file,
                "record_type": "TRANSACTION",
                "payload": json.dumps(row, default=str),
                "reasons": json.dumps(reasons),
                "quarantined_at": now_iso,
            }
            result.rejected_rows.append(quarantine_record)
            for reason in reasons:
                _increment(result.rejection_counts, reason)

    logger.info(
        "validate_chunk: %d valid, %d rejected (batch=%s)",
        len(result.valid_rows),
        len(result.rejected_rows),
        batch_id,
    )
    return result


def write_quarantine(
    rejected_rows: list[dict],
    batch_id: str,
    output_dir: Path,
) -> Path | None:
    """Write rejected rows to a quarantine Parquet file.

    The output file is placed under ``output_dir`` named
    ``quarantine_<batch_id>.parquet``.  If ``rejected_rows`` is empty,
    nothing is written and ``None`` is returned.

    Args:
        rejected_rows: List of quarantine dicts as produced by
            :func:`validate_chunk` (each must have the quarantine schema keys).
        batch_id: Used to name the output file.
        output_dir: Directory where the quarantine parquet is written.
            Created if it does not exist.

    Returns:
        The :class:`~pathlib.Path` of the written file, or ``None`` if there
        were no rejected rows.
    """
    if not rejected_rows:
        logger.debug("No rejected rows; skipping quarantine write (batch=%s)", batch_id)
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"quarantine_{batch_id}.parquet"

    # Build PyArrow table column-by-column (avoids loading into pandas)
    batch_ids = [r["batch_id"] for r in rejected_rows]
    source_files = [r["source_file"] for r in rejected_rows]
    record_types = [r["record_type"] for r in rejected_rows]
    payloads = [r["payload"] for r in rejected_rows]
    reasons_list = [r["reasons"] for r in rejected_rows]
    quarantined_ats = [r["quarantined_at"] for r in rejected_rows]

    table = pa.table(
        {
            "batch_id": pa.array(batch_ids, type=pa.string()),
            "source_file": pa.array(source_files, type=pa.string()),
            "record_type": pa.array(record_types, type=pa.string()),
            "payload": pa.array(payloads, type=pa.string()),
            "reasons": pa.array(reasons_list, type=pa.string()),
            "quarantined_at": pa.array(quarantined_ats, type=pa.string()),
        },
        schema=_QUARANTINE_PA_SCHEMA,
    )

    pq.write_table(table, out_path, compression="snappy")
    logger.info(
        "Quarantine written: %s (%d rows, batch=%s)",
        out_path,
        len(rejected_rows),
        batch_id,
    )
    return out_path
