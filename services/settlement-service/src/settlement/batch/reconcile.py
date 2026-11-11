"""Reconciliation: compare the expected ledger records with the PSP file and classify every gap.

**Matching key** ``(settlement_date, payment_id, record_type)``: the PSP's ``merchant_reference`` is
the payment id. Everything is computed for all dates in one pass.

**Algorithm.**

1. *Number the records of each key.* The PSP records of a key are numbered by ``psp_record_id``. The
   ledger expects ``n`` records for the key (one for a SALE, possibly two for same-day partial
   refunds). The first ``max(n, 1)`` PSP records are *originals*; every further one is a
   ``DUPLICATE_IN_PSP``. For a SALE (``n = 1``) this is the rule "the lowest
   ``psp_record_id`` is the original, the others are duplicates".
2. *Pair originals with expected records.* Within a key both sides are ordered by amount (then by
   id) and paired by rank. Without this, two legitimate same-day refunds of one payment would look
   like a duplicate.
3. *Classify* (full outer join on the key plus the pair rank). Every row ends up in exactly one
   class, so nothing is left "unclassified":

   =====================================  =========================  ============
   situation                              break                      precedence
   =====================================  =========================  ============
   both sides, gross differs              AMOUNT_MISMATCH            over the fee
   both sides, SALE, same gross, fee off  FEE_MISMATCH
   ledger only                            MISSING_IN_PSP
   PSP only, CHARGEBACK                   CHARGEBACK_NOT_IN_LEDGER
   PSP only, anything else                MISSING_IN_LEDGER
   an extra PSP record for a key          DUPLICATE_IN_PSP
   =====================================  =========================  ============

PSP records with a null reference cannot match anything: each is its own orphan. Known limitation:
an item settled on a different date than expected shows up as MISSING_IN_PSP on one date and
MISSING_IN_LEDGER on another (cross-date matching is out of scope).
"""

from dataclasses import dataclass

import polars as pl

from settlement.domain import BreakType, RecordType

KEY = ["settlement_date", "match_ref", "record_type"]

BREAK_COLUMNS = [
    "settlement_date",
    "break_type",
    "payment_id",
    "psp_record_id",
    "ledger_amount_minor",
    "psp_amount_minor",
    "diff_minor",
]
MATCHED_SALE_COLUMNS = [
    "settlement_date",
    "merchant_id",
    "currency",
    "gross_minor",
    "expected_fee_minor",
    "actual_fee_minor",
]


@dataclass(frozen=True)
class Reconciliation:
    breaks: pl.LazyFrame  # BREAK_COLUMNS, one row per discrepancy, not yet numbered
    matched_sales: (
        pl.LazyFrame
    )  # SALEs present on both sides with equal gross (MATCHED_SALE_COLUMNS)


def reconcile(expected: pl.LazyFrame, psp: pl.LazyFrame, payments: pl.LazyFrame) -> Reconciliation:
    """``expected``: ``expected_records``; ``psp``: the PSP file; ``payments``: for known ids."""
    ledger = _number_expected(expected)
    expected_per_key = ledger.group_by(KEY).agg(pl.len().alias("n_expected"))

    # A reference that no payment owns must not name a payment in a break (the generator's rule).
    known = payments.select(
        pl.col("payment_id").alias("known_payment_id"), pl.lit(True).alias("is_known")
    )

    numbered = _number_psp(psp).join(expected_per_key, on=KEY, how="left")
    numbered = numbered.with_columns(pl.col("n_expected").fill_null(0))
    is_duplicate = pl.col("key_index") > pl.max_horizontal(pl.col("n_expected"), pl.lit(1))

    duplicates = numbered.filter(is_duplicate)
    originals = _pair_rank(numbered.filter(~is_duplicate), ["gross_amount_minor", "psp_record_id"])

    joined = ledger.join(originals, on=[*KEY, "pair_rank"], how="full", coalesce=True)
    ledger_present = pl.col("event_id").is_not_null()
    psp_present = pl.col("psp_record_id").is_not_null()
    amounts_differ = pl.col("amount_minor") != pl.col("gross_amount_minor")
    is_sale = pl.col("record_type") == RecordType.SALE.value
    fee_differs = is_sale & (pl.col("expected_fee_minor") != pl.col("fee_minor"))

    classified = joined.with_columns(
        pl.when(ledger_present & psp_present)
        .then(
            pl.when(amounts_differ)
            .then(pl.lit(BreakType.AMOUNT_MISMATCH.value))
            .when(fee_differs)
            .then(pl.lit(BreakType.FEE_MISMATCH.value))
        )
        .when(ledger_present)
        .then(pl.lit(BreakType.MISSING_IN_PSP.value))
        .when(pl.col("record_type") == RecordType.CHARGEBACK.value)
        .then(pl.lit(BreakType.CHARGEBACK_NOT_IN_LEDGER.value))
        .otherwise(pl.lit(BreakType.MISSING_IN_LEDGER.value))
        .alias("break_type")
    )

    matched_sales = classified.filter(
        ledger_present & psp_present & is_sale & ~amounts_differ
    ).select(
        "settlement_date",
        "merchant_id",
        "currency",
        pl.col("amount_minor").alias("gross_minor"),
        "expected_fee_minor",
        pl.col("fee_minor").alias("actual_fee_minor"),
    )

    # Ledger-side amounts: the gross, except FEE_MISMATCH which compares the fees.
    is_fee_break = pl.col("break_type") == BreakType.FEE_MISMATCH.value
    one_sided = classified.filter(pl.col("break_type").is_not_null()).select(
        "settlement_date",
        "break_type",
        _payment_id_expr(ledger_present),
        "psp_record_id",
        pl.when(~ledger_present)
        .then(None)
        .when(is_fee_break)
        .then(pl.col("expected_fee_minor"))
        .otherwise(pl.col("amount_minor"))
        .alias("ledger_amount_minor"),
        pl.when(~psp_present)
        .then(None)
        .when(is_fee_break)
        .then(pl.col("fee_minor"))
        .otherwise(pl.col("gross_amount_minor"))
        .alias("psp_amount_minor"),
    )
    classified_breaks = _with_diff(_resolve_payment_id(one_sided, known))

    duplicate_breaks = _with_diff(
        _resolve_payment_id(
            duplicates.select(
                "settlement_date",
                pl.lit(BreakType.DUPLICATE_IN_PSP.value).alias("break_type"),
                pl.col("merchant_reference").alias("payment_id"),
                "psp_record_id",
                pl.lit(None, dtype=pl.Int64).alias("ledger_amount_minor"),
                pl.col("gross_amount_minor").alias("psp_amount_minor"),
            ),
            known,
        )
    )
    return Reconciliation(
        breaks=pl.concat([classified_breaks, duplicate_breaks]).select(BREAK_COLUMNS),
        matched_sales=matched_sales.select(MATCHED_SALE_COLUMNS),
    )


def _payment_id_expr(ledger_present: pl.Expr) -> pl.Expr:
    """The payment a break is about: the ledger's, else the PSP's reference."""
    return (
        pl.when(ledger_present)
        .then(pl.col("payment_id"))
        .otherwise(pl.col("merchant_reference"))
        .alias("payment_id")
    )


def _resolve_payment_id(frame: pl.LazyFrame, known: pl.LazyFrame) -> pl.LazyFrame:
    """Keep ``payment_id`` only if a payment with that id exists, else null."""
    return (
        frame.join(known, left_on="payment_id", right_on="known_payment_id", how="left")
        .with_columns(
            pl.when(pl.col("is_known").is_not_null()).then(pl.col("payment_id")).alias("payment_id")
        )
        .drop("is_known")
    )


def _with_diff(frame: pl.LazyFrame) -> pl.LazyFrame:
    """``diff_minor`` = PSP minus ledger, a missing side counting as zero."""
    return frame.with_columns(
        (
            pl.col("psp_amount_minor").fill_null(0) - pl.col("ledger_amount_minor").fill_null(0)
        ).alias("diff_minor")
    )


def _number_expected(expected: pl.LazyFrame) -> pl.LazyFrame:
    return _pair_rank(
        expected.with_columns(pl.col("payment_id").alias("match_ref")), ["amount_minor", "event_id"]
    )


def _number_psp(psp: pl.LazyFrame) -> pl.LazyFrame:
    """Number the PSP records of each key by id. A null reference gets a key nobody else has."""
    return (
        psp.with_columns(
            pl.coalesce("merchant_reference", pl.format("~{}", pl.col("psp_record_id"))).alias(
                "match_ref"
            )
        )
        .sort("psp_record_id")
        .with_columns((pl.int_range(pl.len()).over(KEY) + 1).alias("key_index"))
    )


def _pair_rank(frame: pl.LazyFrame, order_by: list[str]) -> pl.LazyFrame:
    """``pair_rank``: 1, 2, ... within each key, in the given order."""
    return frame.sort(order_by).with_columns(
        (pl.int_range(pl.len()).over(KEY) + 1).alias("pair_rank")
    )
