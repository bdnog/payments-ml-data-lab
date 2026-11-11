"""Cross-dataset checks: what no single data contract can express.

Each check returns a list of ``Violation`` (empty means it holds). Together they make the generated
data trustworthy: foreign keys resolve, the labels agree across risk and settlement, the ledger
adds up, and *every* difference between the ledger and the PSP file is an injected break that can
be explained from the data (and nothing else differs).

The "explained" checks are, in effect, a small reference reconciliation. Phase 4 builds the real one
and uses the injected breaks as its answer key.
"""

from datetime import date

import polars as pl

from payments_datagen.generate import generate_world
from payments_datagen.profiles import TINY, Profile
from payments_datagen.settlement import (
    CHARGEBACK_FEE_MINOR,
    EVENT_SETTLEMENT_DAYS,
    expected_fee_minor,
)
from payments_shared.data_contracts import Violation, check_foreign_key

NAME = "invariants"
MAX_EXAMPLES = 3


def _violation(
    kind: str, target: str, bad: pl.DataFrame | pl.Series | int, key: str = ""
) -> list[Violation]:
    """A one-element list when ``bad`` is non-empty, else empty."""
    if isinstance(bad, int):
        count, examples = bad, []
    elif isinstance(bad, pl.Series):
        count, examples = len(bad), [str(v) for v in bad.head(MAX_EXAMPLES).to_list()]
    else:
        count = bad.height
        examples = [str(v) for v in bad[key].head(MAX_EXAMPLES).to_list()] if key else []
    return [Violation(NAME, kind, target, count, examples)] if count else []


def _due_date(merchants: pl.DataFrame, payments: pl.DataFrame) -> pl.DataFrame:
    """Captured payments with the date their sale settles."""
    return (
        payments.filter(pl.col("status") == "CAPTURED")
        .join(merchants.select("merchant_id", "settlement_days", "mdr_bps"), on="merchant_id")
        .with_columns(
            due_date=(
                pl.col("captured_at").dt.date().cast(pl.Datetime("us"))
                + pl.duration(days=pl.col("settlement_days"))
            ).cast(pl.Date)
        )
    )


def check_foreign_keys(d: dict[str, pl.DataFrame]) -> list[Violation]:
    pairs = [
        ("transactions", "account_id", "accounts", "account_id"),
        ("transactions", "merchant_id", "merchants", "merchant_id"),
        ("payments", "transaction_id", "transactions", "transaction_id"),
        ("payments", "merchant_id", "merchants", "merchant_id"),
        ("payments", "account_id", "accounts", "account_id"),
        ("refunds", "payment_id", "payments", "payment_id"),
        ("chargebacks", "payment_id", "payments", "payment_id"),
        ("psp_settlement", "merchant_id", "merchants", "merchant_id"),
        ("injected_breaks", "payment_id", "payments", "payment_id"),
        ("injected_breaks", "psp_record_id", "psp_settlement", "psp_record_id"),
    ]
    found: list[Violation] = []
    for child, child_col, parent, parent_col in pairs:
        found += check_foreign_key(d[child], child_col, d[parent], parent_col)
    found += _violation(
        "one_payment_per_transaction",
        "payments.transaction_id",
        d["payments"].filter(pl.col("transaction_id").is_duplicated())["transaction_id"],
    )
    return found


def check_rows_are_sorted(d: dict[str, pl.DataFrame]) -> list[Violation]:
    order = {
        "merchants": "created_at",
        "accounts": "created_at",
        "transactions": "occurred_at",
        "payments": "authorized_at",
        "refunds": "created_at",
        "chargebacks": "opened_at",
        "psp_settlement": "settlement_date",
        "injected_breaks": "break_id",
    }
    found: list[Violation] = []
    for name, column in order.items():
        if not d[name][column].is_sorted():
            found.append(Violation(NAME, "rows_sorted", f"{name}.{column}", 0))
    return found


def check_labels(d: dict[str, pl.DataFrame]) -> list[Violation]:
    """FRAUD chargebacks and ``chargeback_reported_at`` are two views of the same fact."""
    fraud_cb = (
        d["chargebacks"]
        .filter(pl.col("reason_code") == "FRAUD")
        .join(d["payments"].select("payment_id", "transaction_id"), on="payment_id", how="left")
        .select("transaction_id", "opened_at")
    )
    reported = (
        d["transactions"]
        .filter(pl.col("chargeback_reported_at").is_not_null())
        .select("transaction_id", pl.col("chargeback_reported_at").alias("opened_at"))
    )
    only_chargeback = fraud_cb.join(reported, on=["transaction_id", "opened_at"], how="anti")
    only_label = reported.join(fraud_cb, on=["transaction_id", "opened_at"], how="anti")
    return _violation(
        "fraud_chargeback_matches_label",
        "chargebacks without a matching label",
        only_chargeback,
        "transaction_id",
    ) + _violation(
        "fraud_chargeback_matches_label",
        "labels without a matching chargeback",
        only_label,
        "transaction_id",
    )


def check_ledger(d: dict[str, pl.DataFrame]) -> list[Violation]:
    payments = d["payments"].select("payment_id", "amount_minor", "captured_at", "status")
    refunds = d["refunds"].join(payments, on="payment_id", how="left", suffix="_payment")
    totals = refunds.group_by("payment_id").agg(
        pl.col("amount_minor").sum().alias("refunded"),
        pl.col("amount_minor_payment").first().alias("paid"),
    )
    chargebacks = d["chargebacks"].join(payments, on="payment_id", how="left", suffix="_payment")
    return (
        _violation(
            "refund_within_payment",
            "refund amount above the payment",
            refunds.filter(pl.col("amount_minor") > pl.col("amount_minor_payment")),
            "refund_id",
        )
        + _violation(
            "refund_within_payment",
            "refunds of a payment add up to more than it",
            totals.filter(pl.col("refunded") > pl.col("paid")),
            "payment_id",
        )
        + _violation(
            "refund_after_capture",
            "refund before capture",
            refunds.filter(
                pl.col("captured_at").is_null() | (pl.col("created_at") <= pl.col("captured_at"))
            ),
            "refund_id",
        )
        + _violation(
            "chargeback_on_captured_payment",
            "chargeback on a payment that was not captured",
            chargebacks.filter(pl.col("status") != "CAPTURED"),
            "chargeback_id",
        )
        + _violation(
            "chargeback_full_amount",
            "chargeback amount differs from the payment",
            chargebacks.filter(pl.col("amount_minor") != pl.col("amount_minor_payment")),
            "chargeback_id",
        )
    )


def check_breaks_are_explainable(d: dict[str, pl.DataFrame], last_day: date) -> list[Violation]:
    """Every injected break must be visible in the data, with the stated difference."""
    ledger = d["payments"].select(
        "payment_id", pl.col("amount_minor").alias("ledger_amount"), "status"
    )
    record = d["psp_settlement"].select(
        "psp_record_id",
        "merchant_reference",
        "record_type",
        pl.col("merchant_id").alias("psp_merchant_id"),
        pl.col("gross_amount_minor").alias("psp_gross"),
        pl.col("fee_minor").alias("psp_fee"),
    )
    mdr = d["merchants"].select(pl.col("merchant_id").alias("psp_merchant_id"), "mdr_bps")
    sales = d["psp_settlement"].filter(pl.col("record_type") == "SALE")
    sale_refs = sales["merchant_reference"].drop_nulls()
    sales_per_ref = sales.group_by("merchant_reference").len()
    ledger_chargebacks = d["chargebacks"]["payment_id"]
    known_payments = d["payments"]["payment_id"]

    j = (
        d["injected_breaks"]
        .join(ledger, on="payment_id", how="left")
        .join(record, on="psp_record_id", how="left")
        .join(mdr, on="psp_merchant_id", how="left")
        .join(
            sales_per_ref.rename({"merchant_reference": "payment_id", "len": "sale_count"}),
            on="payment_id",
            how="left",
        )
    )
    diff = pl.col("expected_diff_minor")
    is_sale_of_payment = (pl.col("record_type") == "SALE") & (
        pl.col("merchant_reference") == pl.col("payment_id")
    )
    explained = {
        "AMOUNT_MISMATCH": is_sale_of_payment
        & (diff != 0)
        & (pl.col("psp_gross") - pl.col("ledger_amount") == diff),
        "FEE_MISMATCH": is_sale_of_payment
        & (diff != 0)
        & (pl.col("psp_fee") - expected_fee_minor(pl.col("psp_gross"), pl.col("mdr_bps")) == diff),
        "MISSING_IN_PSP": pl.col("psp_record_id").is_null()
        & (pl.col("status") == "CAPTURED")
        & pl.col("sale_count").is_null()
        & (diff == -pl.col("ledger_amount")),
        "MISSING_IN_LEDGER": (pl.col("record_type") == "SALE")
        & pl.col("payment_id").is_null()
        & (
            pl.col("merchant_reference").is_null()
            | ~pl.col("merchant_reference").is_in(known_payments.implode())
        )
        & (diff == pl.col("psp_gross")),
        "DUPLICATE_IN_PSP": is_sale_of_payment
        & (pl.col("sale_count") >= 2)
        & (diff == pl.col("psp_gross")),
        "CHARGEBACK_NOT_IN_LEDGER": (pl.col("record_type") == "CHARGEBACK")
        & (pl.col("merchant_reference") == pl.col("payment_id"))
        & ~pl.col("payment_id").is_in(ledger_chargebacks.implode())
        & (diff == pl.col("psp_gross")),
    }
    found: list[Violation] = []
    for break_type, condition in explained.items():
        bad = j.filter((pl.col("break_type") == break_type) & ~condition.fill_null(False))
        found += _violation("break_explainable", break_type, bad, "break_id")

    # The reverse for missing sales: every due, captured payment with no sale record is a break.
    due = _due_date(d["merchants"], d["payments"]).filter(pl.col("due_date") <= last_day)
    missing = due.filter(~pl.col("payment_id").is_in(sale_refs.implode()))
    declared = d["injected_breaks"].filter(pl.col("break_type") == "MISSING_IN_PSP")["payment_id"]
    found += _violation(
        "missing_sale_has_break",
        "due payment without a PSP sale and without a MISSING_IN_PSP break",
        missing.filter(~pl.col("payment_id").is_in(declared.implode())),
        "payment_id",
    )
    return found


def _multiset_difference(left: pl.DataFrame, right: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    a = left.group_by(keys).len().rename({"len": "left"})
    b = right.group_by(keys).len().rename({"len": "right"})
    both = a.join(b, on=keys, how="full", coalesce=True).fill_null(0)
    return both.filter(pl.col("left") != pl.col("right"))


def check_nothing_is_unexplained(d: dict[str, pl.DataFrame], last_day: date) -> list[Violation]:
    """Outside the injected breaks, the ledger and the PSP file must agree exactly."""
    breaks = d["injected_breaks"]
    broken_payments = breaks["payment_id"].drop_nulls()
    broken_records = breaks["psp_record_id"].drop_nulls()
    psp = d["psp_settlement"]
    cutoff = pl.lit(last_day)

    # Sales: one record per due captured payment, with the ledger amount and the contracted fee.
    due = _due_date(d["merchants"], d["payments"]).filter(
        (pl.col("due_date") <= cutoff) & ~pl.col("payment_id").is_in(broken_payments.implode())
    )
    sales = (
        psp.filter(pl.col("record_type") == "SALE")
        .group_by("merchant_reference")
        .agg(
            pl.len().alias("n"),
            pl.col("gross_amount_minor").first().alias("gross"),
            pl.col("fee_minor").first().alias("fee"),
        )
    )
    checked = due.join(sales, left_on="payment_id", right_on="merchant_reference", how="left")
    wrong_sales = checked.filter(
        (pl.col("n") != 1)
        | pl.col("n").is_null()
        | (pl.col("gross") != pl.col("amount_minor"))
        | (pl.col("fee") != expected_fee_minor(pl.col("amount_minor"), pl.col("mdr_bps")))
    )
    # Sale records that no payment explains must all be MISSING_IN_LEDGER breaks.
    orphan_sales = psp.filter(
        (pl.col("record_type") == "SALE")
        & (
            pl.col("merchant_reference").is_null()
            | ~pl.col("merchant_reference").is_in(d["payments"]["payment_id"].implode())
        )
        & ~pl.col("psp_record_id").is_in(broken_records.implode())
    )

    # Refunds and chargebacks: the due ones in the ledger are exactly the ones in the file.
    event_days = EVENT_SETTLEMENT_DAYS

    def due_event(frame: pl.DataFrame, at: str) -> pl.DataFrame:
        settles = (
            pl.col(at).dt.date().cast(pl.Datetime("us")) + pl.duration(days=event_days)
        ).cast(pl.Date)
        return frame.filter(settles <= cutoff)

    ledger_refunds = due_event(d["refunds"], "created_at").select("payment_id", "amount_minor")
    psp_refunds = psp.filter(pl.col("record_type") == "REFUND").select(
        pl.col("merchant_reference").alias("payment_id"),
        pl.col("gross_amount_minor").alias("amount_minor"),
    )
    ledger_cbs = due_event(d["chargebacks"], "opened_at").select("payment_id", "amount_minor")
    psp_cbs = psp.filter(
        (pl.col("record_type") == "CHARGEBACK")
        & ~pl.col("psp_record_id").is_in(broken_records.implode())
    ).select(
        pl.col("merchant_reference").alias("payment_id"),
        pl.col("gross_amount_minor").alias("amount_minor"),
    )
    return (
        _violation(
            "unexplained_sale",
            "due sale that differs from the ledger without a break",
            wrong_sales,
            "payment_id",
        )
        + _violation(
            "unexplained_sale",
            "PSP sale without a payment and without a break",
            orphan_sales,
            "psp_record_id",
        )
        + _violation(
            "unexplained_refund",
            "refunds differ between ledger and PSP file",
            _multiset_difference(ledger_refunds, psp_refunds, ["payment_id", "amount_minor"]),
            "payment_id",
        )
        + _violation(
            "unexplained_chargeback",
            "chargebacks differ between ledger and PSP file",
            _multiset_difference(ledger_cbs, psp_cbs, ["payment_id", "amount_minor"]),
            "payment_id",
        )
        + _violation(
            "chargeback_fee",
            "PSP chargeback with a fee other than the flat one",
            psp.filter(
                (pl.col("record_type") == "CHARGEBACK")
                & (pl.col("fee_minor") != CHARGEBACK_FEE_MINOR)
            ),
            "psp_record_id",
        )
    )


def check_dataset_invariants(d: dict[str, pl.DataFrame], last_day: date) -> list[Violation]:
    """All invariants that only need the written datasets."""
    return (
        check_foreign_keys(d)
        + check_rows_are_sorted(d)
        + check_labels(d)
        + check_ledger(d)
        + check_breaks_are_explainable(d, last_day)
        + check_nothing_is_unexplained(d, last_day)
    )


def check_determinism(profile: Profile = TINY) -> list[Violation]:
    """Same seed gives equal frames; a different seed gives different ones."""
    first = generate_world(profile)
    again = generate_world(profile)
    other = generate_world(profile.with_overrides(seed=profile.seed + 1))
    found = [
        Violation(NAME, "determinism", f"{name}: same seed, different frames", 0)
        for name, frame in first.frames.items()
        if not frame.equals(again.frames[name])
    ]
    if first["transactions"].equals(other["transactions"]):
        found.append(Violation(NAME, "determinism", "different seed, identical transactions", 0))
    return found
