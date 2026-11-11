from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from risk_scoring.training.labels import (
    DEFAULT_LABEL_MATURITY_DAYS,
    ImmatureLabelsError,
    check_label_maturity,
    derive_label,
    training_as_of,
    training_label,
)
from risk_scoring.training.splits import DEFAULT_FRACTIONS, SPLIT_NAMES, plan_split

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def frame(*chargebacks: datetime | None) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "occurred_at": [T0] * len(chargebacks),
            "chargeback_reported_at": list(chargebacks),
        },
        schema={
            "occurred_at": pl.Datetime("us", "UTC"),
            "chargeback_reported_at": pl.Datetime("us", "UTC"),
        },
    )


# --- labels -----------------------------------------------------------------------------------


def test_label_is_a_chargeback_reported_on_or_before_as_of():
    as_of = T0 + timedelta(days=10)
    df = frame(None, T0 + timedelta(days=3), as_of, as_of + timedelta(microseconds=1))

    assert derive_label(df, as_of).to_list() == [False, True, True, False]


def test_eventual_label_is_any_chargeback():
    df = frame(None, T0 + timedelta(days=400))
    assert derive_label(df).to_list() == [False, True]


def test_label_never_looks_at_the_ground_truth_columns():
    df = frame(None).with_columns(pl.lit(True).alias("is_fraud"))
    assert derive_label(df, T0 + timedelta(days=99)).to_list() == [False]


def test_training_as_of_is_train_end_plus_the_maturity():
    assert training_as_of(T0, 45) == T0 + timedelta(days=45)


@pytest.mark.parametrize("days_waited", [0, 1, 44])
def test_guard_refuses_immature_labels(days_waited):
    with pytest.raises(ImmatureLabelsError, match="45 days"):
        check_label_maturity(T0, T0 + timedelta(days=days_waited), DEFAULT_LABEL_MATURITY_DAYS)


def test_guard_accepts_exactly_the_maturity_or_more():
    check_label_maturity(T0, T0 + timedelta(days=45))
    check_label_maturity(T0, T0 + timedelta(days=90))


def test_training_label_applies_the_guard():
    df = frame(T0 + timedelta(days=5))
    with pytest.raises(ImmatureLabelsError):
        training_label(df, train_end=T0, as_of=T0 + timedelta(days=10))
    assert training_label(df, train_end=T0, as_of=T0 + timedelta(days=45)).to_list() == [True]


def test_a_too_early_as_of_loses_fraud_labels(training_transactions):
    """The reason for the guard: reading labels too soon silently turns fraud into "legitimate"."""
    plan = plan_split(training_transactions)
    train = plan.apply(training_transactions)["train"]
    mature = plan.train_end + timedelta(days=DEFAULT_LABEL_MATURITY_DAYS)

    eventual = derive_label(train).sum()
    at_maturity = derive_label(train, mature).sum()
    too_early = derive_label(train, plan.train_end).sum()

    assert at_maturity == eventual, "after the maturity period no training label is missing"
    assert too_early < 0.8 * eventual, "reading labels at train_end loses many fraud labels"


def test_generator_never_reports_a_chargeback_later_than_the_maturity(training_transactions):
    delay = training_transactions.filter(pl.col("chargeback_reported_at").is_not_null()).select(
        (pl.col("chargeback_reported_at") - pl.col("occurred_at")).max().alias("max_delay")
    )["max_delay"][0]
    assert delay <= timedelta(days=DEFAULT_LABEL_MATURITY_DAYS)


# --- splits -----------------------------------------------------------------------------------


def test_splits_are_disjoint_ordered_and_cover_the_window(training_transactions):
    plan = plan_split(training_transactions)
    parts = plan.apply(training_transactions)

    assert tuple(parts) == SPLIT_NAMES
    assert sum(p.height for p in parts.values()) == training_transactions.height
    ids = [i for p in parts.values() for i in p["transaction_id"].to_list()]
    assert len(set(ids)) == len(ids), "a transaction is in two splits"
    assert (
        parts["train"]["occurred_at"].max()
        < parts["validation"]["occurred_at"].min()
        <= parts["validation"]["occurred_at"].max()
        < parts["test"]["occurred_at"].min()
    )
    assert plan.window_start <= training_transactions["occurred_at"].min()
    assert plan.window_end > training_transactions["occurred_at"].max()


def test_full_profile_fractions_cut_at_days_99_and_126_of_180():
    days = 180
    df = pl.DataFrame(
        {"occurred_at": [T0, T0 + timedelta(days=days - 1, hours=23)]},
        schema={"occurred_at": pl.Datetime("us", "UTC")},
    )
    plan = plan_split(df)

    assert (plan.train_end - plan.window_start).days == 99
    assert (plan.validation_end - plan.window_start).days == 126
    assert (plan.window_end - plan.window_start).days == 180
    assert DEFAULT_FRACTIONS == (0.55, 0.15, 0.30)


def test_split_boundary_belongs_to_the_later_split():
    df = pl.DataFrame(
        {"occurred_at": [T0, T0 + timedelta(days=99), T0 + timedelta(days=179)]},
        schema={"occurred_at": pl.Datetime("us", "UTC")},
    )
    parts = plan_split(df).apply(df)
    assert [p.height for p in parts.values()] == [1, 1, 1]


@pytest.mark.parametrize("fractions", [(0.5, 0.5, 0.5), (0.6, 0.4, 0.0), (0.5, 0.5)])
def test_invalid_fractions_are_rejected(fractions, training_transactions):
    with pytest.raises(ValueError, match="fractions"):
        plan_split(training_transactions, fractions)  # type: ignore[arg-type]


def test_cannot_plan_an_empty_frame(training_transactions):
    with pytest.raises(ValueError, match="empty"):
        plan_split(training_transactions.head(0))
