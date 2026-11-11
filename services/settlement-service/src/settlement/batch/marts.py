"""Builders of the three marts, as lazy frames over the expected records and the reconciliation.

* ``daily_settlement_by_merchant``: the **ledger-expected payout**, i.e. what the platform owes each
  merchant for a settlement date. It is not what the PSP confirmed: that is what the breaks are for.
  ``net = gross - fee - refunds - chargebacks``; all components are non-negative magnitudes and the
  net can be negative. The flat chargeback fee is absorbed by the platform, so it is not deducted.
* ``fee_summary``: only SALEs present on both sides with equal gross, so a non-zero difference
  isolates fee errors (``FEE_MISMATCH``) from amount errors.
* ``reconciliation_breaks``: one row per discrepancy; ``run_date`` is the date being reconciled,
  which equals ``settlement_date`` because a run reconciles what is due on its date.
"""

from datetime import date

import polars as pl

from settlement.domain import RecordType

EPOCH = date(1970, 1, 1)
MAX_BREAKS_PER_DATE = 0xFFFF


def build_daily_settlement(expected: pl.LazyFrame) -> pl.LazyFrame:
    amount = pl.col("amount_minor")
    kind = pl.col("record_type")
    sale, refund, chargeback = (
        RecordType.SALE.value,
        RecordType.REFUND.value,
        RecordType.CHARGEBACK.value,
    )
    return (
        expected.group_by("settlement_date", "merchant_id", "currency")
        .agg(
            (kind == sale).sum().cast(pl.Int64).alias("payment_count"),
            amount.filter(kind == sale).sum().alias("gross_minor"),
            pl.col("expected_fee_minor").filter(kind == sale).sum().alias("fee_minor"),
            amount.filter(kind == refund).sum().alias("refunds_minor"),
            amount.filter(kind == chargeback).sum().alias("chargebacks_minor"),
        )
        .with_columns(
            (
                pl.col("gross_minor")
                - pl.col("fee_minor")
                - pl.col("refunds_minor")
                - pl.col("chargebacks_minor")
            ).alias("net_minor")
        )
    )


def build_fee_summary(matched_sales: pl.LazyFrame) -> pl.LazyFrame:
    return (
        matched_sales.group_by("settlement_date", "merchant_id", "currency")
        .agg(
            pl.col("gross_minor").sum(),
            pl.col("expected_fee_minor").sum(),
            pl.col("actual_fee_minor").sum(),
        )
        .with_columns(
            (pl.col("actual_fee_minor") - pl.col("expected_fee_minor")).alias("fee_diff_minor")
        )
    )


def break_id(settlement_date: date, sequence: int) -> str:
    """``brk_`` + 4 hex digits of the day number + 4 hex digits of the sequence within the day.

    Deterministic (a rerun gives the same ids), unique across dates, and sorted like the rows. It
    fits the contract pattern ``^brk_[0-9a-f]{8}$`` with no random or hashed part to collide.
    """
    day = (settlement_date - EPOCH).days
    if not 0 <= day <= 0xFFFF:
        raise ValueError(f"settlement date {settlement_date} is outside the range of break ids")
    if not 0 <= sequence <= MAX_BREAKS_PER_DATE:
        raise ValueError(f"more than {MAX_BREAKS_PER_DATE + 1} breaks on {settlement_date}")
    return f"brk_{day:04x}{sequence:04x}"


def build_breaks_mart(breaks: pl.DataFrame) -> pl.DataFrame:
    """Number the breaks of each date and add ``run_date``. Eager: ids are built in Python."""
    ordered = breaks.sort(
        ["settlement_date", "break_type", "payment_id", "psp_record_id"],
        nulls_last=True,
    )
    sequence = ordered.select(pl.int_range(pl.len()).over("settlement_date").alias("n"))["n"]
    ids = [
        break_id(day, int(n)) for day, n in zip(ordered["settlement_date"], sequence, strict=True)
    ]
    return ordered.with_columns(
        pl.Series("break_id", ids, dtype=pl.String),
        pl.col("settlement_date").alias("run_date"),
    )
