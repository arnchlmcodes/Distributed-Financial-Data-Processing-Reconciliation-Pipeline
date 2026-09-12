"""
schema.py — Explicit PySpark schemas for all data sources.
No inferSchema. All money fields use DecimalType(18,2).
"""
from pyspark.sql.types import (
    DecimalType,
    IntegerType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# Raw PaySim CSV schema (amounts come in as strings first, then cast)
PAYSIM_RAW_SCHEMA = StructType([
    StructField("step", IntegerType(), nullable=False),
    StructField("type", StringType(), nullable=False),
    StructField("amount", StringType(), nullable=False),  # read as string, cast to Decimal later
    StructField("nameOrig", StringType(), nullable=False),
    StructField("oldbalanceOrg", StringType(), nullable=True),
    StructField("newbalanceOrig", StringType(), nullable=True),
    StructField("nameDest", StringType(), nullable=False),
    StructField("oldbalanceDest", StringType(), nullable=True),
    StructField("newbalanceDest", StringType(), nullable=True),
    StructField("isFraud", ShortType(), nullable=False),
    StructField("isFlaggedFraud", ShortType(), nullable=False),
])

# Enriched transactions schema (after M2 enrichment)
TRANSACTION_SCHEMA = StructType([
    StructField("transaction_id", StringType(), nullable=False),
    StructField("batch_id", StringType(), nullable=False),
    StructField("step", IntegerType(), nullable=False),
    StructField("type", StringType(), nullable=False),
    StructField("amount", DecimalType(18, 2), nullable=False),
    StructField("name_orig", StringType(), nullable=False),
    StructField("old_balance_orig", DecimalType(18, 2), nullable=True),
    StructField("new_balance_orig", DecimalType(18, 2), nullable=True),
    StructField("name_dest", StringType(), nullable=False),
    StructField("old_balance_dest", DecimalType(18, 2), nullable=True),
    StructField("new_balance_dest", DecimalType(18, 2), nullable=True),
    StructField("is_fraud", ShortType(), nullable=False),
    StructField("is_flagged_fraud", ShortType(), nullable=False),
    StructField("currency", StringType(), nullable=False),
    StructField("fx_pair", StringType(), nullable=True),
    StructField("transaction_timestamp", TimestampType(), nullable=False),
])

# Settlement schema
SETTLEMENT_SCHEMA = StructType([
    StructField("settlement_id", StringType(), nullable=False),
    StructField("transaction_id", StringType(), nullable=False),
    StructField("batch_id", StringType(), nullable=False),
    StructField("settled_amount", DecimalType(18, 2), nullable=False),
    StructField("settlement_currency", StringType(), nullable=False),
    StructField("settlement_timestamp", TimestampType(), nullable=True),
    StructField("status", StringType(), nullable=False),
])

# FX rates schema
FX_RATE_SCHEMA = StructType([
    StructField("currency_pair", StringType(), nullable=False),
    StructField("rate", DecimalType(18, 6), nullable=False),
    StructField("effective_timestamp", TimestampType(), nullable=False),
])

# Quarantine schema
QUARANTINE_SCHEMA = StructType([
    StructField("batch_id", StringType(), nullable=False),
    StructField("source_file", StringType(), nullable=False),
    StructField("record_type", StringType(), nullable=False),
    StructField("payload", StringType(), nullable=False),
    StructField("reasons", StringType(), nullable=False),  # JSON list
    StructField("quarantined_at", TimestampType(), nullable=False),
])
