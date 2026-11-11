"""The invariants must hold on generated data and must *fail* when the data is corrupted.

A check that never fails proves nothing, so every invariant has a mutation test here.
"""

import polars as pl
import pytest

from payments_datagen.invariants import check_dataset_invariants, check_determinism
from payments_datagen.profiles import TINY


@pytest.fixture(scope="module")
def last_day(stats_world):
    return stats_world.universe.profile.last_day


def kinds(frames, last_day) -> set[str]:
    return {v.kind for v in check_dataset_invariants(frames, last_day)}


def mutated(world, name: str, change) -> dict[str, pl.DataFrame]:
    frames = dict(world.frames)
    frames[name] = change(frames[name])
    return frames


def test_generated_data_satisfies_every_invariant(stats_world, last_day):
    assert check_dataset_invariants(stats_world.frames, last_day) == []


def test_sample_profile_satisfies_every_invariant(sample_world):
    assert (
        check_dataset_invariants(sample_world.frames, sample_world.universe.profile.last_day) == []
    )


def test_determinism_check_passes():
    assert check_determinism(TINY) == []


def test_foreign_key_orphan_is_caught(stats_world, last_day):
    frames = mutated(
        stats_world,
        "psp_settlement",
        lambda f: f.with_columns(
            merchant_id=pl.when(pl.int_range(pl.len()) == 0)
            .then(pl.lit("mer_ffffff"))
            .otherwise(pl.col("merchant_id"))
        ),
    )
    assert "foreign_key" in kinds(frames, last_day)


def test_two_payments_for_one_transaction_are_caught(stats_world, last_day):
    frames = mutated(
        stats_world,
        "payments",
        lambda f: f.with_columns(
            transaction_id=pl.when(pl.int_range(pl.len()) == 1)
            .then(pl.col("transaction_id").first())
            .otherwise(pl.col("transaction_id"))
        ),
    )
    assert "one_payment_per_transaction" in kinds(frames, last_day)


def test_unsorted_rows_are_caught(stats_world, last_day):
    frames = mutated(stats_world, "refunds", lambda f: f.reverse())
    assert "rows_sorted" in kinds(frames, last_day)


def test_fraud_chargeback_that_disagrees_with_the_label_is_caught(stats_world, last_day):
    frames = mutated(
        stats_world,
        "chargebacks",
        lambda f: f.with_columns(
            opened_at=pl.when(pl.col("reason_code") == "FRAUD")
            .then(pl.col("opened_at") + pl.duration(hours=1))
            .otherwise(pl.col("opened_at"))
        ),
    )
    assert "fraud_chargeback_matches_label" in kinds(frames, last_day)


def test_refund_above_the_payment_amount_is_caught(stats_world, last_day):
    frames = mutated(
        stats_world, "refunds", lambda f: f.with_columns(amount_minor=pl.col("amount_minor") * 100)
    )
    assert "refund_within_payment" in kinds(frames, last_day)


def test_refund_before_capture_is_caught(stats_world, last_day):
    frames = mutated(
        stats_world,
        "refunds",
        lambda f: f.with_columns(created_at=pl.col("created_at") - pl.duration(days=30)),
    )
    assert "refund_after_capture" in kinds(frames, last_day)


def test_chargeback_with_the_wrong_amount_is_caught(stats_world, last_day):
    frames = mutated(
        stats_world,
        "chargebacks",
        lambda f: f.with_columns(amount_minor=pl.col("amount_minor") + 1),
    )
    assert "chargeback_full_amount" in kinds(frames, last_day)


def test_a_break_whose_difference_is_wrong_is_not_explainable(stats_world, last_day):
    frames = mutated(
        stats_world,
        "injected_breaks",
        lambda f: f.with_columns(
            expected_diff_minor=pl.when(pl.col("break_type") == "AMOUNT_MISMATCH")
            .then(pl.col("expected_diff_minor") + 1)
            .otherwise(pl.col("expected_diff_minor"))
        ),
    )
    assert "break_explainable" in kinds(frames, last_day)


@pytest.mark.parametrize(
    "break_type",
    [
        "AMOUNT_MISMATCH",
        "FEE_MISMATCH",
        "MISSING_IN_PSP",
        "MISSING_IN_LEDGER",
        "DUPLICATE_IN_PSP",
        "CHARGEBACK_NOT_IN_LEDGER",
    ],
)
def test_each_break_type_is_checked_individually(stats_world, last_day, break_type):
    frames = mutated(
        stats_world,
        "injected_breaks",
        lambda f: f.with_columns(
            expected_diff_minor=pl.when(pl.col("break_type") == break_type)
            .then(pl.col("expected_diff_minor") + 7)
            .otherwise(pl.col("expected_diff_minor"))
        ),
    )
    violations = check_dataset_invariants(frames, last_day)
    assert [v.target for v in violations if v.kind == "break_explainable"] == [break_type]


def test_a_missing_sale_without_its_break_is_caught(stats_world, last_day):
    frames = mutated(
        stats_world,
        "injected_breaks",
        lambda f: f.filter(pl.col("break_type") != "MISSING_IN_PSP"),
    )
    assert "missing_sale_has_break" in kinds(frames, last_day)


def test_a_psp_amount_that_differs_without_a_break_is_caught(stats_world, last_day):
    breaks = stats_world["injected_breaks"]
    victim = stats_world["psp_settlement"].filter(
        (pl.col("record_type") == "SALE")
        & ~pl.col("psp_record_id").is_in(breaks["psp_record_id"].drop_nulls().implode())
        & pl.col("merchant_reference").is_not_null()
    )["psp_record_id"][0]
    frames = mutated(
        stats_world,
        "psp_settlement",
        lambda f: f.with_columns(
            gross_amount_minor=pl.when(pl.col("psp_record_id") == victim)
            .then(pl.col("gross_amount_minor") + 1)
            .otherwise(pl.col("gross_amount_minor"))
        ),
    )
    assert "unexplained_sale" in kinds(frames, last_day)


def test_a_silently_duplicated_sale_is_caught(stats_world, last_day):
    psp = stats_world["psp_settlement"]
    breaks = stats_world["injected_breaks"]
    clean = psp.filter(
        (pl.col("record_type") == "SALE")
        & ~pl.col("psp_record_id").is_in(breaks["psp_record_id"].drop_nulls().implode())
        & pl.col("merchant_reference").is_not_null()
    ).head(1)
    extra = clean.with_columns(psp_record_id=pl.lit("psp_ffffffffffff"))
    frames = mutated(stats_world, "psp_settlement", lambda f: pl.concat([f, extra]))
    assert "unexplained_sale" in kinds(frames, last_day)


def test_a_dropped_refund_record_is_caught(stats_world, last_day):
    frames = mutated(
        stats_world,
        "psp_settlement",
        lambda f: f.filter(
            pl.col("psp_record_id")
            != f.filter(pl.col("record_type") == "REFUND")["psp_record_id"][0]
        ),
    )
    assert "unexplained_refund" in kinds(frames, last_day)


def test_a_dropped_chargeback_record_is_caught(stats_world, last_day):
    breaks = stats_world["injected_breaks"]
    ledger_chargeback_record = stats_world["psp_settlement"].filter(
        (pl.col("record_type") == "CHARGEBACK")
        & ~pl.col("psp_record_id").is_in(breaks["psp_record_id"].drop_nulls().implode())
    )["psp_record_id"][0]
    frames = mutated(
        stats_world,
        "psp_settlement",
        lambda f: f.filter(pl.col("psp_record_id") != ledger_chargeback_record),
    )
    assert "unexplained_chargeback" in kinds(frames, last_day)
