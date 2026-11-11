"""The ledger side: what the platform expects the PSP to settle, record by record.

One row per expected PSP record, for every date at once (the functions are date-agnostic: they
group by ``settlement_date`` and never take a "run date"). The generator is the reference for which
records exist:

* a **SALE** for each ``CAPTURED`` payment (voided payments moved no money), due on the capture date
  plus the merchant's ``settlement_days``;
* a **REFUND** for each refund and a **CHARGEBACK** for each chargeback, due the day after the
  event.

The ``expected_fee_minor`` of a SALE is the contracted fee on the ledger amount; a REFUND has
none; a CHARGEBACK carries the flat PSP fee, which is *not* compared and not deducted from the
merchant.
"""

import polars as pl

from settlement.batch.expressions import expected_fee_expr, settlement_date_expr
from settlement.domain import CHARGEBACK_FEE_MINOR, EVENT_SETTLEMENT_DAYS, RecordType
from settlement.storage.datasets import InputDatasets

EXPECTED_COLUMNS = [
    "settlement_date",
    "merchant_id",
    "payment_id",
    "record_type",
    "event_id",  # the id of the ledger row behind the record (payment, refund or chargeback)
    "currency",
    "amount_minor",
    "expected_fee_minor",
]


def expected_records(inputs: InputDatasets) -> pl.LazyFrame:
    payments = inputs.payments
    merchants = inputs.merchants.select("merchant_id", "mdr_bps", "settlement_days")
    owner = payments.select("payment_id", "merchant_id", "currency")

    sales = (
        payments.filter(pl.col("status") == "CAPTURED")
        .join(merchants, on="merchant_id", how="left")
        .select(
            settlement_date_expr(pl.col("captured_at"), pl.col("settlement_days")).alias(
                "settlement_date"
            ),
            "merchant_id",
            "payment_id",
            pl.lit(RecordType.SALE.value).alias("record_type"),
            pl.col("payment_id").alias("event_id"),
            "currency",
            "amount_minor",
            expected_fee_expr(pl.col("amount_minor"), pl.col("mdr_bps")).alias(
                "expected_fee_minor"
            ),
        )
    )
    refunds = inputs.refunds.join(owner, on="payment_id", how="left").select(
        settlement_date_expr(pl.col("created_at"), EVENT_SETTLEMENT_DAYS).alias("settlement_date"),
        "merchant_id",
        "payment_id",
        pl.lit(RecordType.REFUND.value).alias("record_type"),
        pl.col("refund_id").alias("event_id"),
        "currency",
        "amount_minor",
        pl.lit(0, dtype=pl.Int64).alias("expected_fee_minor"),
    )
    chargebacks = inputs.chargebacks.join(owner, on="payment_id", how="left").select(
        settlement_date_expr(pl.col("opened_at"), EVENT_SETTLEMENT_DAYS).alias("settlement_date"),
        "merchant_id",
        "payment_id",
        pl.lit(RecordType.CHARGEBACK.value).alias("record_type"),
        pl.col("chargeback_id").alias("event_id"),
        "currency",
        "amount_minor",
        pl.lit(CHARGEBACK_FEE_MINOR, dtype=pl.Int64).alias("expected_fee_minor"),
    )
    return pl.concat([sales, refunds, chargebacks]).select(EXPECTED_COLUMNS)
