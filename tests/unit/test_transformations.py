"""
test_transformations.py — Unit tests for PySpark transformations.
"""
from __future__ import annotations

from datetime import UTC
from decimal import Decimal

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DecimalType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from src.transformations.fx_transform import apply_fx_rates
from src.transformations.settlement_transform import (
    detect_issues,
    match_settlements,
)
from src.transformations.transaction_transform import (
    cast_money_columns,
    deduplicate_transactions,
)


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    spark_session = (
        SparkSession.builder.master("local[1]")
        .appName("PipelineUnitTests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark_session.sparkContext.setLogLevel("ERROR")
    yield spark_session
    spark_session.stop()


class TestTransactionTransform:
    def test_cast_money_columns(self, spark: SparkSession):
        schema = StructType([
            StructField("transaction_id", StringType(), False),
            StructField("amount", StringType(), False),
            StructField("old_balance_orig", StringType(), True),
        ])
        data = [("tx-1", "123.456", "1000.00")]
        df = spark.createDataFrame(data, schema)
        transformed = cast_money_columns(df)

        assert dict(transformed.dtypes)["amount"] == "decimal(18,2)"
        assert dict(transformed.dtypes)["old_balance_orig"] == "decimal(18,2)"

    def test_deduplicate_exact_duplicates(self, spark: SparkSession):
        schema = StructType([
            StructField("transaction_id", StringType(), False),
            StructField("amount", StringType(), False),
            StructField("type", StringType(), False),
        ])
        # Two identical rows
        data = [
            ("tx-1", "100.00", "PAYMENT"),
            ("tx-1", "100.00", "PAYMENT"),
        ]
        df = spark.createDataFrame(data, schema)
        clean_df, quarantine_df = deduplicate_transactions(df, batch_id="b1", spark=spark)

        assert clean_df.count() == 1
        assert quarantine_df.count() == 0

    def test_deduplicate_conflicting_duplicates(self, spark: SparkSession):
        schema = StructType([
            StructField("transaction_id", StringType(), False),
            StructField("amount", StringType(), False),
            StructField("type", StringType(), False),
        ])
        # Two conflicting rows with same ID
        data = [
            ("tx-1", "100.00", "PAYMENT"),
            ("tx-1", "500.00", "TRANSFER"),
        ]
        df = spark.createDataFrame(data, schema)
        clean_df, quarantine_df = deduplicate_transactions(df, batch_id="b1", spark=spark)

        assert clean_df.count() == 0
        assert quarantine_df.count() == 2


class TestFxTransform:
    def test_apply_fx_rates(self, spark: SparkSession):
        txn_schema = StructType([
            StructField("transaction_id", StringType(), False),
            StructField("amount", DecimalType(18, 2), False),
            StructField("fx_pair", StringType(), True),
            StructField("transaction_timestamp", TimestampType(), False),
        ])
        fx_schema = StructType([
            StructField("currency_pair", StringType(), False),
            StructField("rate", DecimalType(18, 6), False),
            StructField("effective_timestamp", TimestampType(), False),
        ])

        from datetime import datetime
        txn_data = [
            ("tx-1", Decimal("100.00"), "EUR/USD", datetime(2023, 1, 2, 12, 0, tzinfo=UTC)),
            ("tx-2", Decimal("1000.00"), "INR/USD", datetime(2023, 1, 2, 12, 0, tzinfo=UTC)),
            ("tx-3", Decimal("250.50"), "USD/USD", datetime(2023, 1, 2, 12, 0, tzinfo=UTC)),
            ("tx-4", Decimal("50.00"), "UNKNOWN/USD", datetime(2023, 1, 2, 12, 0, tzinfo=UTC)),
            ("tx-5", Decimal("75.00"), "EUR/USD", datetime(2022, 12, 31, 12, 0, tzinfo=UTC)),
        ]
        fx_data = [
            ("EUR/USD", Decimal("1.100000"), datetime(2023, 1, 1, 0, 0, tzinfo=UTC)),
            ("EUR/USD", Decimal("1.200000"), datetime(2023, 1, 2, 0, 0, tzinfo=UTC)),
            ("INR/USD", Decimal("0.011976"), datetime(2023, 1, 1, 0, 0, tzinfo=UTC)),
            ("USD/USD", Decimal("1.000000"), datetime(2023, 1, 1, 0, 0, tzinfo=UTC)),
        ]
        txn_df = spark.createDataFrame(txn_data, txn_schema)
        fx_df = spark.createDataFrame(fx_data, fx_schema)

        result_df = apply_fx_rates(txn_df, fx_df)
        rows = {r["transaction_id"]: r for r in result_df.collect()}

        # tx-1: 100 EUR * 1.200000 = 120.00 USD
        assert rows["tx-1"]["fx_rate"] == Decimal("1.200000")
        assert rows["tx-1"]["amount_usd"] == Decimal("120.00")
        assert rows["tx-1"]["fx_issue"] is None

        # tx-2: 1000.00 INR * 0.011976 = 11.98 USD
        assert rows["tx-2"]["fx_rate"] == Decimal("0.011976")
        assert rows["tx-2"]["amount_usd"] == Decimal("11.98")
        assert rows["tx-2"]["fx_issue"] is None

        # tx-3: USD identity: 250.50 USD * 1.000000 = 250.50 USD
        assert rows["tx-3"]["fx_rate"] == Decimal("1.000000")
        assert rows["tx-3"]["amount_usd"] == Decimal("250.50")
        assert rows["tx-3"]["fx_issue"] is None

        # tx-4: unknown currency pair -> MISSING_FX_RATE
        assert rows["tx-4"]["fx_rate"] is None
        assert rows["tx-4"]["amount_usd"] is None
        assert rows["tx-4"]["fx_issue"] == "MISSING_FX_RATE"

        # tx-5: transaction timestamp pre-dates earliest rate -> MISSING_FX_RATE
        assert rows["tx-5"]["fx_rate"] is None
        assert rows["tx-5"]["amount_usd"] is None
        assert rows["tx-5"]["fx_issue"] == "MISSING_FX_RATE"


class TestSettlementTransform:
    def test_settlement_matching_and_issues(self, spark: SparkSession):
        from datetime import datetime
        txn_schema = StructType([
            StructField("transaction_id", StringType(), False),
            StructField("amount", DecimalType(18, 2), False),
            StructField("type", StringType(), False),
            StructField("step", IntegerType(), False),
            StructField("currency", StringType(), False),
            StructField("transaction_timestamp", TimestampType(), False),
        ])
        stl_schema = StructType([
            StructField("settlement_id", StringType(), False),
            StructField("transaction_id", StringType(), False),
            StructField("batch_id", StringType(), False),
            StructField("settled_amount", DecimalType(18, 2), False),
            StructField("settlement_currency", StringType(), False),
            StructField("settlement_timestamp", TimestampType(), True),
            StructField("status", StringType(), False),
        ])

        txn_data = [
            ("tx-1", Decimal("100.00"), "PAYMENT", 1, "USD", datetime(2023, 1, 1, 1, 0, tzinfo=UTC)),
            ("tx-2", Decimal("200.00"), "PAYMENT", 2, "USD", datetime(2023, 1, 1, 2, 0, tzinfo=UTC)),
            ("tx-3", Decimal("300.00"), "PAYMENT", 3, "USD", datetime(2023, 1, 1, 3, 0, tzinfo=UTC)),
        ]
        stl_data = [
            ("s-1", "tx-1", "b1", Decimal("100.00"), "USD", datetime(2023, 1, 1, 2, 0, tzinfo=UTC), "COMPLETED"),
            # tx-2 has AMOUNT_MISMATCH
            ("s-2", "tx-2", "b1", Decimal("190.00"), "USD", datetime(2023, 1, 1, 3, 0, tzinfo=UTC), "COMPLETED"),
            # tx-3 has no settlement -> MISSING_SETTLEMENT
        ]
        txn_df = spark.createDataFrame(txn_data, txn_schema)
        stl_df = spark.createDataFrame(stl_data, stl_schema)

        matched = match_settlements(txn_df, stl_df)
        assert matched.count() == 3

        issues = detect_issues(matched, batch_id="b1", tolerance=Decimal("0.01"))
        issue_types = {r["issue_type"] for r in issues.collect()}
        assert "MISSING_SETTLEMENT" in issue_types
        assert "AMOUNT_MISMATCH" in issue_types
