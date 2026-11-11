from datetime import timedelta

import polars as pl
import pytest

from payments_datagen.fraud import (
    CHARGEBACK_DELAY_DAYS,
    PATTERN_SHARES,
    apply_outcomes,
    fraud_budgets,
    generate_fraud,
)
from payments_datagen.profiles import SAMPLE, Profile
from payments_datagen.transactions import assemble_transactions, generate_legit
from payments_datagen.universe import build_universe
from payments_shared.data_contracts import assert_valid, load_contract

# Bigger than the committed sample so statistical checks are tight, still well under a second.
STATS = Profile(
    name="stats",
    n_accounts=1_500,
    n_merchants=40,
    days=60,
    target_transactions=40_000,
    fraud_rate=0.02,
    break_rate=0.05,
)


def build(profile):
    universe = build_universe(profile)
    legit = generate_legit(universe)
    raw_fraud = generate_fraud(universe, legit)
    txns = assemble_transactions(universe, legit, raw_fraud)
    labeled, outcomes = apply_outcomes(universe, txns)
    return universe, labeled, outcomes


@pytest.fixture(scope="module")
def sample():
    return build(SAMPLE)


@pytest.fixture(scope="module")
def stats():
    return build(STATS)


def episodes(txns: pl.DataFrame, pattern: str) -> pl.DataFrame:
    """One row per episode, over its online transactions.

    Every episode uses a device of its own, so the device identifies it. In-person (pos) rows carry
    a shared terminal id instead of a device, so they cannot be attributed to an episode this way.
    """
    return (
        txns.filter((pl.col("fraud_pattern") == pattern) & (pl.col("channel") != "pos"))
        .group_by("device_id")
        .agg(
            n=pl.len(),
            accounts=pl.col("account_id").n_unique(),
            span=pl.col("occurred_at").max() - pl.col("occurred_at").min(),
        )
    )


def test_output_honors_the_transactions_contract(sample):
    _, txns, _ = sample
    assert_valid(txns, load_contract("transactions"))


def test_realized_fraud_rate_is_close_to_the_profile(stats):
    _, txns, _ = stats
    assert txns["is_fraud"].mean() == pytest.approx(STATS.fraud_rate, abs=0.003)


def test_every_pattern_is_present_in_the_sample_profile(sample):
    _, txns, _ = sample
    found = set(txns.filter(pl.col("is_fraud"))["fraud_pattern"])
    assert found == set(PATTERN_SHARES)


def test_pattern_mix_follows_the_shares(stats):
    _, txns, _ = stats
    fraud = txns.filter(pl.col("is_fraud"))
    for pattern, share in PATTERN_SHARES.items():
        realized = (fraud["fraud_pattern"] == pattern).mean()
        assert realized == pytest.approx(share, abs=0.02)


def test_budgets_give_every_pattern_at_least_one_transaction():
    assert all(v >= 1 for v in fraud_budgets(10, 0.01).values())
    assert set(fraud_budgets(10, 0.0).values()) == {0}


def test_card_testing_is_many_tiny_charges_from_one_device_in_minutes(stats):
    _, txns, _ = stats
    eps = episodes(txns, "CARD_TESTING")
    assert (eps["accounts"] == 1).all()
    assert (eps["span"] <= timedelta(minutes=15)).all()
    assert eps["n"].max() <= 30
    card_testing = txns.filter(pl.col("fraud_pattern") == "CARD_TESTING")
    assert card_testing["amount_minor"].median() <= 500
    assert (card_testing["channel"] != "pos").all()


def test_account_takeover_hits_old_accounts_from_a_new_device(stats):
    universe, txns, _ = stats
    ato = txns.filter(pl.col("fraud_pattern") == "ACCOUNT_TAKEOVER")
    assert (ato["account_age_days"] > 90).all()
    eps = episodes(txns, "ACCOUNT_TAKEOVER")
    assert (eps["span"] <= timedelta(hours=2)).all()
    assert eps["n"].max() <= 5
    first_per_episode = ato.sort("occurred_at").group_by("device_id", maintain_order=True).first()
    assert first_per_episode["device_is_new_for_account"].all()
    with_history = ato.filter(pl.col("avg_amount_30d_minor").is_not_null())
    assert (with_history["amount_minor"] > 1.5 * with_history["avg_amount_30d_minor"]).mean() > 0.7
    risky_mccs = {"5732", "5944", "5816", "5967", "4829"}
    assert set(ato["mcc"]) <= risky_mccs


def test_geo_mismatch_is_a_brazilian_card_abroad_on_a_new_account(stats):
    _, txns, _ = stats
    geo = txns.filter(pl.col("fraud_pattern") == "GEO_MISMATCH")
    assert (geo["issuer_country"] == "BR").all()
    assert (geo["account_age_days"] < 30).all()
    assert (geo["ip_country"] != "BR").mean() > 0.8
    assert geo["is_proxy"].mean() > 0.8


def test_velocity_burst_is_many_purchases_in_an_hour(stats):
    _, txns, _ = stats
    eps = episodes(txns, "VELOCITY_BURST")
    assert (eps["span"] <= timedelta(hours=1)).all()
    assert eps["n"].max() <= 20
    burst = txns.filter(pl.col("fraud_pattern") == "VELOCITY_BURST")
    assert burst["txn_count_1h"].max() >= 7


def test_classes_overlap_so_the_problem_is_not_trivially_separable(stats):
    _, txns, _ = stats
    fraud = txns.filter(pl.col("is_fraud") & (pl.col("channel") != "pos"))
    legit = txns.filter(~pl.col("is_fraud") & (pl.col("channel") != "pos"))
    stealthy = fraud.filter(
        (pl.col("ip_country") == pl.col("issuer_country")) & ~pl.col("is_proxy")
    )
    assert stealthy.height > 0
    assert (legit["ip_country"] != legit["issuer_country"]).any()
    assert legit["is_proxy"].any()


def test_declines_are_hidden_and_follow_the_rates(stats):
    _, txns, outcomes = stats
    assert "declined" not in txns.columns
    assert outcomes["transaction_id"].to_list() == txns["transaction_id"].to_list()
    joined = txns.select("transaction_id", "is_fraud").join(outcomes, on="transaction_id")
    legit_rate = joined.filter(~pl.col("is_fraud"))["declined"].mean()
    fraud_rate = joined.filter(pl.col("is_fraud"))["declined"].mean()
    approved = joined.filter(~pl.col("declined"))
    assert legit_rate == pytest.approx(0.06, abs=0.015)
    assert fraud_rate == pytest.approx(0.40, abs=0.06)
    assert approved["voided"].mean() == pytest.approx(0.03, abs=0.015)
    assert not joined.filter(pl.col("declined"))["voided"].any()


def test_only_approved_captured_fraud_can_be_charged_back_and_about_85_percent_is(stats):
    _, txns, outcomes = stats
    joined = txns.join(outcomes, on="transaction_id")
    reported = joined.filter(pl.col("chargeback_reported_at").is_not_null())
    assert reported["is_fraud"].all()
    assert not reported["declined"].any()
    assert not reported["voided"].any()
    eligible = joined.filter(pl.col("is_fraud") & ~pl.col("declined") & ~pl.col("voided"))
    assert reported.height / eligible.height == pytest.approx(0.85, abs=0.05)


def test_chargebacks_arrive_7_to_45_days_later_some_after_the_window(stats):
    universe, txns, _ = stats
    reported = txns.filter(pl.col("chargeback_reported_at").is_not_null())
    delay = (
        reported["chargeback_reported_at"] - reported["occurred_at"]
    ).dt.total_seconds() / 86_400
    low, high = CHARGEBACK_DELAY_DAYS
    assert delay.min() >= low
    assert delay.max() <= high
    assert (reported["chargeback_reported_at"] >= universe.profile.window_end).any()


def test_label_delay_means_the_training_label_depends_on_the_cutoff(stats):
    universe, txns, _ = stats
    fraud = txns.filter(pl.col("is_fraud"))

    def labeled_by(cutoff_days: int) -> int:
        cutoff = universe.profile.window_start + timedelta(days=cutoff_days)
        return fraud.filter(pl.col("chargeback_reported_at") <= cutoff).height

    assert labeled_by(15) < labeled_by(40) < labeled_by(universe.profile.days) < fraud.height


def test_generation_is_deterministic():
    _, first, first_outcomes = build(SAMPLE)
    _, again, again_outcomes = build(SAMPLE)
    assert first.equals(again)
    assert first_outcomes.equals(again_outcomes)


def test_no_fraud_when_the_rate_is_zero():
    _, txns, _ = build(SAMPLE.with_overrides(fraud_rate=0.0))
    assert not txns["is_fraud"].any()
    assert txns["chargeback_reported_at"].null_count() == txns.height
