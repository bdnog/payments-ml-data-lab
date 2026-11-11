"""Settlement inputs, derived from the very same transactions as the risk data.

The chain is: approved transactions become **payments** (the ledger); some payments get **refunds**
and **chargebacks**; the PSP then reports what it settled in the **psp_settlement** file; finally a
known set of discrepancies is **injected** into that file and recorded in **injected_breaks**, so
the Phase 4 reconciliation has a ground truth to be tested against.

Domain rules encoded here (the settlement domain must reproduce them):

* **Fee rounding.** The expected SALE fee is ``round_half_up(gross * mdr_bps / 10000)``, computed
  with integer arithmetic as ``(gross * mdr_bps + 5000) // 10000``. There are no floats anywhere.
* **Due-date cutoff.** A record only exists in the PSP file when its ``settlement_date`` is on or
  before the last day of the data window. A captured payment that is not due yet has no PSP record
  and is *not* a break.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from payments_datagen.rng import stream
from payments_datagen.universe import Universe
from payments_datagen.util import (
    US_PER_DAY,
    US_PER_HOUR,
    epoch_us_of,
    hex_ids,
    ts_series,
)

CAPTURE_MAX_DAYS = 2
REFUND_RATE = 0.04  # of captured payments
SECOND_REFUND_SHARE = 0.15  # of refunded payments get a second partial refund
FULL_REFUND_SHARE = 0.60  # of single refunds
FRIENDLY_CHARGEBACK_RATE = 0.002  # of legitimate captured payments
CHARGEBACK_FEE_MINOR = 1_500  # flat, per chargeback
REFUND_REASONS = ["CUSTOMER_REQUEST", "DUPLICATE", "PRODUCT_ISSUE"]
FRIENDLY_REASONS = ["PRODUCT_NOT_RECEIVED", "DUPLICATE_CHARGE"]
EVENT_SETTLEMENT_DAYS = 1  # refunds and chargebacks settle the day after the event

BREAK_TYPES = [
    "AMOUNT_MISMATCH",
    "MISSING_IN_PSP",
    "MISSING_IN_LEDGER",
    "DUPLICATE_IN_PSP",
    "FEE_MISMATCH",
    "CHARGEBACK_NOT_IN_LEDGER",
]
BREAK_WEIGHTS = np.array([0.25, 0.20, 0.15, 0.15, 0.15, 0.10])
CHARGEBACK_MIN_DAYS = 10  # an injected PSP chargeback lands 10 to 40 days after capture
CHARGEBACK_MAX_DAYS = 40

PSP_COLUMNS = [
    "merchant_reference",
    "record_type",
    "merchant_id",
    "gross_amount_minor",
    "fee_minor",
    "net_amount_minor",
    "currency",
    "settlement_date",
]


def expected_fee_minor(gross_minor: pl.Expr, mdr_bps: pl.Expr) -> pl.Expr:
    """Contracted fee: ``round_half_up(gross * mdr_bps / 10000)`` in integer arithmetic."""
    return (gross_minor * mdr_bps + 5_000) // 10_000


@dataclass(frozen=True)
class SettlementData:
    payments: pl.DataFrame
    refunds: pl.DataFrame
    chargebacks: pl.DataFrame
    psp_settlement: pl.DataFrame
    injected_breaks: pl.DataFrame


def _add_days(dates: pl.Expr, days: int | pl.Expr) -> pl.Expr:
    return (dates.cast(pl.Datetime("us")) + pl.duration(days=days)).cast(pl.Date)


def build_payments(
    universe: Universe, txns: pl.DataFrame, outcomes: pl.DataFrame, rng
) -> pl.DataFrame:
    """One payment per approved transaction; about 3% are voided before capture."""
    approved = txns.join(outcomes, on="transaction_id", how="inner", maintain_order="left").filter(
        ~pl.col("declined")
    )
    n = approved.height
    voided = approved["voided"].to_numpy()
    authorized_us = epoch_us_of(approved["occurred_at"])
    capture_delay = rng.integers(0, CAPTURE_MAX_DAYS * US_PER_DAY, size=n)
    captured_at = pl.when(pl.Series(~voided)).then(
        ts_series("captured_at", authorized_us + capture_delay)
    )
    ids = hex_ids(stream(universe.profile.seed, "pay_ids"), "pay_", 12, n, "payment_id")
    return approved.with_columns(
        ids,
        authorized_at=pl.col("occurred_at"),
        status=pl.when(pl.Series(voided)).then(pl.lit("VOIDED")).otherwise(pl.lit("CAPTURED")),
        captured_at=captured_at,
    ).select(
        "payment_id",
        "transaction_id",
        "merchant_id",
        "account_id",
        "amount_minor",
        "currency",
        "status",
        "authorized_at",
        "captured_at",
    )


def build_chargebacks(
    universe: Universe, payments: pl.DataFrame, txns: pl.DataFrame, rng
) -> pl.DataFrame:
    """FRAUD chargebacks mirror the labels in ``transactions``, plus a few friendly-fraud ones."""
    ledger = payments.select(
        "payment_id", "transaction_id", "amount_minor", "status", "captured_at"
    )
    fraud = (
        txns.filter(pl.col("chargeback_reported_at").is_not_null())
        .select("transaction_id", pl.col("chargeback_reported_at").alias("opened_at"))
        .join(ledger, on="transaction_id", how="inner")
    )
    if fraud.height and (fraud["status"] != "CAPTURED").any():
        raise ValueError("a fraud chargeback points at a payment that was never captured")
    fraud = fraud.select("payment_id", "amount_minor", "opened_at", reason_code=pl.lit("FRAUD"))

    legit = ledger.join(
        txns.select("transaction_id", "is_fraud"), on="transaction_id", how="inner"
    ).filter((pl.col("status") == "CAPTURED") & ~pl.col("is_fraud"))
    friendly = legit.filter(pl.Series(rng.random(legit.height) < FRIENDLY_CHARGEBACK_RATE))
    n = friendly.height
    opened_us = epoch_us_of(friendly["captured_at"]) + rng.integers(
        10 * US_PER_DAY, 60 * US_PER_DAY, size=n
    )
    friendly = friendly.select(
        "payment_id",
        "amount_minor",
        ts_series("opened_at", opened_us),
        reason_code=pl.Series(FRIENDLY_REASONS).gather(rng.integers(0, len(FRIENDLY_REASONS), n)),
    )

    combined = pl.concat([fraud, friendly]).sort(["opened_at", "payment_id"])
    ids = hex_ids(
        stream(universe.profile.seed, "cbk_ids"), "cbk_", 12, combined.height, "chargeback_id"
    )
    return combined.with_columns(ids).select(
        "chargeback_id", "payment_id", "amount_minor", "opened_at", "reason_code"
    )


def build_refunds(
    universe: Universe, payments: pl.DataFrame, chargebacks: pl.DataFrame, rng
) -> pl.DataFrame:
    """Refunds on about 4% of captured payments, never on charged-back ones.

    Most refunded payments get one refund (60% of them for the full amount); 15% get two partial
    refunds that add up to less than the payment amount.
    """
    eligible = payments.filter(
        (pl.col("status") == "CAPTURED")
        & ~pl.col("payment_id").is_in(chargebacks["payment_id"].implode())
    )
    chosen = eligible.filter(pl.Series(rng.random(eligible.height) < REFUND_RATE))
    m = chosen.height
    amount = chosen["amount_minor"].to_numpy()
    two = rng.random(m) < SECOND_REFUND_SHARE
    full = (rng.random(m) < FULL_REFUND_SHARE) & ~two

    partial = np.maximum(1, (amount * rng.uniform(0.1, 0.9, m)).astype(np.int64))
    first_of_two = np.maximum(1, (amount * rng.uniform(0.2, 0.5, m)).astype(np.int64))
    first = np.where(two, first_of_two, np.where(full, amount, partial))
    second = np.maximum(1, ((amount - first) * rng.uniform(0.3, 1.0, m)).astype(np.int64))

    first_at = epoch_us_of(chosen["captured_at"]) + rng.integers(US_PER_HOUR, 20 * US_PER_DAY, m)
    second_at = first_at + rng.integers(US_PER_HOUR, 10 * US_PER_DAY, m)
    reasons = pl.Series(REFUND_REASONS)

    def rows(mask: np.ndarray, amounts: np.ndarray, at: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "payment_id": chosen["payment_id"].filter(pl.Series(mask)),
                "amount_minor": amounts[mask],
                "created_at": ts_series("created_at", at[mask]),
                "reason": reasons.gather(rng.integers(0, len(reasons), int(mask.sum()))),
            }
        )

    combined = pl.concat(
        [rows(np.ones(m, dtype=bool), first, first_at), rows(two, second, second_at)]
    ).sort(["created_at", "payment_id"])
    ids = hex_ids(
        stream(universe.profile.seed, "ref_ids"), "ref_", 12, combined.height, "refund_id"
    )
    return combined.with_columns(ids).select(
        "refund_id", "payment_id", "amount_minor", "created_at", "reason"
    )


def _psp_rows(
    reference: pl.Expr,
    record_type: str,
    merchant_id: pl.Expr,
    gross: pl.Expr,
    fee: pl.Expr,
    net: pl.Expr,
    settlement_date: pl.Expr,
) -> list[pl.Expr]:
    return [
        reference.alias("merchant_reference"),
        pl.lit(record_type).alias("record_type"),
        merchant_id.alias("merchant_id"),
        gross.cast(pl.Int64).alias("gross_amount_minor"),
        fee.cast(pl.Int64).alias("fee_minor"),
        net.cast(pl.Int64).alias("net_amount_minor"),
        pl.lit("BRL").alias("currency"),
        settlement_date.alias("settlement_date"),
    ]


def build_psp_records(
    universe: Universe, payments: pl.DataFrame, refunds: pl.DataFrame, chargebacks: pl.DataFrame
) -> pl.DataFrame:
    """What a flawless PSP would report: every due sale, refund and chargeback, and nothing else.

    The frame also carries the internal columns ``_mdr_bps`` and ``_captured_date`` for sales, which
    the break injection needs and which are dropped before anything is written.
    """
    last_day = universe.profile.last_day
    merchants = universe.merchants.select("merchant_id", "mdr_bps", "settlement_days")
    sales = (
        payments.filter(pl.col("status") == "CAPTURED")
        .join(merchants, on="merchant_id", how="left", maintain_order="left")
        .with_columns(
            fee=expected_fee_minor(pl.col("amount_minor"), pl.col("mdr_bps")),
            captured_date=pl.col("captured_at").dt.date(),
        )
        .select(
            *_psp_rows(
                pl.col("payment_id"),
                "SALE",
                pl.col("merchant_id"),
                pl.col("amount_minor"),
                pl.col("fee"),
                pl.col("amount_minor") - pl.col("fee"),
                _add_days(pl.col("captured_date"), pl.col("settlement_days")),
            ),
            pl.col("mdr_bps").alias("_mdr_bps"),
            pl.col("captured_date").alias("_captured_date"),
        )
    )
    by_payment = payments.select("payment_id", "merchant_id")
    refund_rows = refunds.join(
        by_payment, on="payment_id", how="left", maintain_order="left"
    ).select(
        *_psp_rows(
            pl.col("payment_id"),
            "REFUND",
            pl.col("merchant_id"),
            pl.col("amount_minor"),
            pl.lit(0),
            -pl.col("amount_minor"),
            _add_days(pl.col("created_at").dt.date(), EVENT_SETTLEMENT_DAYS),
        )
    )
    chargeback_rows = chargebacks.join(
        by_payment, on="payment_id", how="left", maintain_order="left"
    ).select(
        *_psp_rows(
            pl.col("payment_id"),
            "CHARGEBACK",
            pl.col("merchant_id"),
            pl.col("amount_minor"),
            pl.lit(CHARGEBACK_FEE_MINOR),
            -(pl.col("amount_minor") + CHARGEBACK_FEE_MINOR),
            _add_days(pl.col("opened_at").dt.date(), EVENT_SETTLEMENT_DAYS),
        )
    )
    internal = [
        pl.lit(None, pl.Int64).alias("_mdr_bps"),
        pl.lit(None, pl.Date).alias("_captured_date"),
    ]
    records = pl.concat(
        [sales, refund_rows.with_columns(internal), chargeback_rows.with_columns(internal)]
    )
    return records.filter(pl.col("settlement_date") <= last_day)


def _break_counts(rng: np.random.Generator, total: int) -> dict[str, int]:
    return dict(
        zip(BREAK_TYPES, (int(c) for c in rng.multinomial(total, BREAK_WEIGHTS)), strict=True)
    )


def inject_breaks(
    universe: Universe,
    payments: pl.DataFrame,
    refunds: pl.DataFrame,
    chargebacks: pl.DataFrame,
    psp: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Corrupt the PSP file in known ways. Returns ``(psp_with_break_marks, break_rows)``.

    Each victim payment gets at most one break. Victims of the different types are drawn from one
    shuffled pool without replacement. ``psp_with_break_marks`` has a ``_break`` column (the
    sequence number of the break a record belongs to) and ``break_rows`` is keyed by that number;
    both are resolved to real ids once the PSP records get theirs.
    """
    profile = universe.profile
    rng = stream(profile.seed, "breaks")
    sales = psp.filter(pl.col("record_type") == "SALE")
    others = psp.filter(pl.col("record_type") != "SALE").drop("_mdr_bps", "_captured_date")
    n_sales = sales.height

    n_captured = payments.filter(pl.col("status") == "CAPTURED").height
    counts = _break_counts(rng, round(profile.break_rate * n_captured))

    gross = sales["gross_amount_minor"].to_numpy().copy()
    fee = sales["fee_minor"].to_numpy().copy()
    mdr = sales["_mdr_bps"].to_numpy()
    reference = sales["merchant_reference"]
    due_date = sales["settlement_date"]
    captured_date = sales["_captured_date"].to_numpy()
    mark = np.full(n_sales, -1, dtype=np.int64)  # break sequence number per sale record

    # Victim selection: a chargeback missing from the ledger needs a clean, old-enough payment; all
    # types draw from what is left, so no payment is picked twice.
    has_other_events = reference.is_in(
        pl.concat([refunds["payment_id"], chargebacks["payment_id"]]).implode()
    ).to_numpy()
    last_day = np.datetime64(profile.last_day, "D")
    old_enough = captured_date <= last_day - np.timedelta64(CHARGEBACK_MIN_DAYS, "D")
    free = np.ones(n_sales, dtype=bool)

    def draw(eligible: np.ndarray, wanted: int) -> np.ndarray:
        candidates = np.flatnonzero(eligible & free)
        chosen = rng.permutation(candidates)[:wanted]
        free[chosen] = False
        return chosen

    cb_idx = draw(~has_other_events & old_enough, counts["CHARGEBACK_NOT_IN_LEDGER"])
    amount_idx = draw(np.ones(n_sales, bool), counts["AMOUNT_MISMATCH"])
    missing_idx = draw(np.ones(n_sales, bool), counts["MISSING_IN_PSP"])
    duplicate_idx = draw(np.ones(n_sales, bool), counts["DUPLICATE_IN_PSP"])
    fee_idx = draw(np.ones(n_sales, bool), counts["FEE_MISMATCH"])

    seq = 0
    break_parts: list[pl.DataFrame] = []

    def record(
        kind: str, payment_ids: pl.Series, expected: np.ndarray, dates: pl.Series
    ) -> np.ndarray:
        nonlocal seq
        numbers = np.arange(seq, seq + len(expected), dtype=np.int64)
        seq += len(expected)
        break_parts.append(
            pl.DataFrame(
                {
                    "seq": numbers,
                    "break_type": pl.Series([kind] * len(expected), dtype=pl.String),
                    "payment_id": payment_ids,
                    "expected_diff_minor": expected.astype(np.int64),
                    "_date": dates,
                }
            )
        )
        return numbers

    # AMOUNT_MISMATCH: the PSP settled a different gross (1% to 20% off); its fee follows its gross.
    ledger = gross[amount_idx]
    delta = np.maximum(1, np.rint(ledger * rng.uniform(0.01, 0.20, len(ledger)))).astype(np.int64)
    delta *= rng.choice([-1, 1], size=len(ledger))
    changed = np.where(ledger + delta < 1, ledger + np.abs(delta), ledger + delta)
    gross[amount_idx] = changed
    fee[amount_idx] = (changed * mdr[amount_idx] + 5_000) // 10_000
    mark[amount_idx] = record(
        "AMOUNT_MISMATCH",
        reference.gather(amount_idx),
        changed - ledger,
        due_date.gather(amount_idx),
    )

    # FEE_MISMATCH: the PSP charged a fee other than the contracted one.
    contracted = fee[fee_idx]
    magnitude = rng.integers(1, np.maximum(2, gross[fee_idx] // 20) + 1)
    shifted = contracted + magnitude * rng.choice([-1, 1], size=len(contracted))
    shifted = np.where(shifted < 0, contracted + magnitude, shifted)
    fee[fee_idx] = shifted
    mark[fee_idx] = record(
        "FEE_MISMATCH", reference.gather(fee_idx), shifted - contracted, due_date.gather(fee_idx)
    )

    # MISSING_IN_PSP: a due sale is absent from the file.
    missing_marks = record(
        "MISSING_IN_PSP",
        reference.gather(missing_idx),
        -gross[missing_idx],
        due_date.gather(missing_idx),
    )
    mark[missing_idx] = missing_marks
    keep = np.ones(n_sales, dtype=bool)
    keep[missing_idx] = False

    # DUPLICATE_IN_PSP: the same sale is reported twice; the extra copy carries the mark.
    duplicate_marks = record(
        "DUPLICATE_IN_PSP",
        reference.gather(duplicate_idx),
        gross[duplicate_idx],
        due_date.gather(duplicate_idx),
    )

    sales_out = sales.drop("_mdr_bps", "_captured_date").with_columns(
        gross_amount_minor=pl.Series(gross),
        fee_minor=pl.Series(fee),
    )
    sales_out = sales_out.with_columns(
        net_amount_minor=pl.col("gross_amount_minor") - pl.col("fee_minor"),
        _break=pl.when(pl.Series(mark >= 0)).then(pl.Series(mark)).otherwise(None),
    )
    duplicates = (
        sales_out.gather(duplicate_idx)
        .with_columns(_break=pl.Series(duplicate_marks))
        .select(sales_out.columns)
    )
    sales_out = sales_out.filter(pl.Series(keep))

    # CHARGEBACK_NOT_IN_LEDGER: the PSP reports a chargeback the ledger never recorded.
    span = np.minimum(CHARGEBACK_MAX_DAYS, (last_day - captured_date[cb_idx]).astype(int))
    offset = CHARGEBACK_MIN_DAYS + (
        rng.random(len(cb_idx)) * (span - CHARGEBACK_MIN_DAYS + 1)
    ).astype(int)
    cb_date = pl.Series(captured_date[cb_idx] + offset.astype("timedelta64[D]"))
    cb_gross = gross[cb_idx]
    cb_marks = record("CHARGEBACK_NOT_IN_LEDGER", reference.gather(cb_idx), cb_gross, cb_date)
    cb_rows = pl.DataFrame(
        {
            "merchant_reference": reference.gather(cb_idx),
            "record_type": ["CHARGEBACK"] * len(cb_idx),
            "merchant_id": sales["merchant_id"].gather(cb_idx),
            "gross_amount_minor": cb_gross,
            "fee_minor": np.full(len(cb_idx), CHARGEBACK_FEE_MINOR, dtype=np.int64),
            "net_amount_minor": -(cb_gross + CHARGEBACK_FEE_MINOR),
            "currency": ["BRL"] * len(cb_idx),
            "settlement_date": cb_date,
            "_break": cb_marks,
        },
        schema_overrides={"record_type": pl.String, "currency": pl.String},
    )

    # MISSING_IN_LEDGER: orphan sales, with a null or unknown reference, that no payment explains.
    orphans = _orphan_sales(universe, payments, counts["MISSING_IN_LEDGER"], rng)
    orphan_marks = record(
        "MISSING_IN_LEDGER",
        pl.Series([None] * orphans.height, dtype=pl.String),
        orphans["gross_amount_minor"].to_numpy(),
        orphans["settlement_date"],
    )
    orphans = orphans.with_columns(_break=pl.Series(orphan_marks))

    columns = [*PSP_COLUMNS, "_break"]
    psp_out = pl.concat(
        [
            sales_out.select(columns),
            others.with_columns(_break=pl.lit(None, pl.Int64)).select(columns),
            duplicates.select(columns),
            cb_rows.select(columns),
            orphans.select(columns),
        ]
    )
    return psp_out, pl.concat(break_parts)


def _orphan_sales(universe: Universe, payments: pl.DataFrame, count: int, rng) -> pl.DataFrame:
    profile = universe.profile
    merchants = universe.merchants
    pick = rng.integers(0, merchants.height, size=count)
    gross = np.clip(np.rint(rng.lognormal(np.log(10_000), 1.0, size=count)), 50, 5_000_000)
    gross = gross.astype(np.int64)
    mdr = merchants["mdr_bps"].to_numpy()[pick]
    fee = (gross * mdr + 5_000) // 10_000

    # Half have no reference at all; half have a well-formed id that no payment owns.
    unknown = hex_ids(rng, "pay_", 12, count, "merchant_reference")
    collides = unknown.is_in(payments["payment_id"].implode()).to_numpy()
    no_reference = (rng.random(count) < 0.5) | collides
    reference = unknown.to_frame().select(
        pl.when(pl.Series(no_reference))
        .then(pl.lit(None, pl.String))
        .otherwise(pl.col("merchant_reference"))
        .alias("merchant_reference")
    )["merchant_reference"]

    day = rng.integers(0, profile.days, size=count).astype("timedelta64[D]")
    settlement_date = pl.Series(np.datetime64(profile.start_date, "D") + day)
    return pl.DataFrame(
        {
            "merchant_reference": reference,
            "record_type": pl.Series(["SALE"] * count, dtype=pl.String),
            "merchant_id": merchants["merchant_id"].gather(pick),
            "gross_amount_minor": gross,
            "fee_minor": fee,
            "net_amount_minor": gross - fee,
            "currency": pl.Series(["BRL"] * count, dtype=pl.String),
            "settlement_date": settlement_date,
        }
    )


def finalize_psp_and_breaks(
    universe: Universe, psp: pl.DataFrame, breaks: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Give PSP records and breaks their ids, now that the file's final order is known."""
    seed = universe.profile.seed
    psp = psp.sort(
        [
            "settlement_date",
            "merchant_id",
            "record_type",
            "merchant_reference",
            "gross_amount_minor",
            "_break",
        ],
        nulls_last=[False, False, False, True, False, False],
    )
    psp = psp.with_columns(
        hex_ids(stream(seed, "psp_ids"), "psp_", 12, psp.height, "psp_record_id")
    )
    breaks = breaks.join(
        psp.select("psp_record_id", "_break").drop_nulls("_break"),
        left_on="seq",
        right_on="_break",
        how="left",
        maintain_order="left",
    ).sort(["_date", "break_type", "payment_id", "seq"], nulls_last=[False, False, True, False])
    breaks = breaks.with_columns(
        hex_ids(stream(seed, "brk_ids"), "brk_", 8, breaks.height, "break_id")
    ).select("break_id", "break_type", "payment_id", "psp_record_id", "expected_diff_minor")
    psp = psp.select("psp_record_id", *PSP_COLUMNS)
    return psp, breaks


def build_settlement(
    universe: Universe, txns: pl.DataFrame, outcomes: pl.DataFrame
) -> SettlementData:
    """Derive every settlement dataset from the transactions and their hidden outcomes."""
    rng = stream(universe.profile.seed, "settlement")
    payments = build_payments(universe, txns, outcomes, rng)
    chargebacks = build_chargebacks(universe, payments, txns, rng)
    refunds = build_refunds(universe, payments, chargebacks, rng)
    base = build_psp_records(universe, payments, refunds, chargebacks)
    marked, break_rows = inject_breaks(universe, payments, refunds, chargebacks, base)
    psp, breaks = finalize_psp_and_breaks(universe, marked, break_rows)
    return SettlementData(payments, refunds, chargebacks, psp, breaks)
