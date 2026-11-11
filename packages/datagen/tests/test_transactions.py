from datetime import timedelta

import polars as pl
import pytest

from payments_datagen.profiles import SAMPLE, TINY
from payments_datagen.transactions import (
    add_point_in_time_features,
    assemble_transactions,
    generate_legit,
)
from payments_datagen.universe import build_universe
from payments_shared.data_contracts import assert_valid, check_foreign_key, load_contract


@pytest.fixture(scope="module")
def tiny():
    universe = build_universe(TINY)
    legit = generate_legit(universe)
    return universe, legit, assemble_transactions(universe, legit)


@pytest.fixture(scope="module")
def sample():
    universe = build_universe(SAMPLE)
    legit = generate_legit(universe)
    return universe, legit, assemble_transactions(universe, legit)


def test_output_honors_the_transactions_contract(tiny):
    _, _, txns = tiny
    assert_valid(txns, load_contract("transactions"))


def test_foreign_keys_resolve(tiny):
    universe, _, txns = tiny
    assert check_foreign_key(txns, "account_id", universe.accounts, "account_id") == []
    assert check_foreign_key(txns, "merchant_id", universe.merchants, "merchant_id") == []


def test_rows_and_ids_are_in_time_order(tiny):
    _, _, txns = tiny
    assert txns["occurred_at"].is_sorted()
    assert txns["transaction_id"].is_sorted()


def test_all_transactions_fall_inside_the_window_and_after_account_creation(sample):
    universe, _, txns = sample
    assert txns["occurred_at"].min() >= universe.profile.window_start
    assert txns["occurred_at"].max() < universe.profile.window_end
    assert (txns["account_age_days"] >= 0).all()


def test_volume_is_close_to_the_target(sample):
    _, _, txns = sample
    expected = SAMPLE.target_transactions * (1 - SAMPLE.fraud_rate)
    assert abs(txns.height - expected) / expected < 0.12


def test_generation_is_deterministic(tiny):
    universe, _, txns = tiny
    again = assemble_transactions(universe, generate_legit(build_universe(TINY)))
    assert txns.equals(again)


def test_traffic_has_an_evening_peak_and_a_quiet_night(sample):
    _, _, txns = sample
    hours = txns["occurred_at"].dt.hour()
    evening = ((hours >= 19) & (hours <= 23)).sum()
    night = (hours <= 4).sum()
    assert evening > 2 * night


def test_in_person_purchases_have_no_ip_signals(sample):
    _, _, txns = sample
    pos = txns.filter(pl.col("channel") == "pos")
    assert pos.height > 0
    assert pos["ip_country"].null_count() == pos.height
    assert not pos["is_proxy"].any()
    assert not pos["device_is_new_for_account"].any()
    assert txns.filter(pl.col("channel") != "pos")["ip_country"].null_count() == 0


def test_hard_negatives_are_present_and_unlabeled_as_fraud(sample):
    _, legit, txns = sample
    assert set(legit["scenario"]) == {"normal", "traveler", "big_ticket", "burst"}
    assert not txns["is_fraud"].any()
    travelers = legit.filter(pl.col("scenario") == "traveler")
    assert not travelers["is_proxy"].any()
    assert (travelers["channel"] != "pos").all()


def test_new_accounts_with_normal_behavior_exist(sample):
    _, _, txns = sample
    young = txns.filter(pl.col("account_age_days") < 30)
    assert young.height > 0


def test_ip_country_is_mostly_the_issuer_country_for_legit_online_traffic(sample):
    _, _, txns = sample
    online = txns.filter(pl.col("channel") != "pos")
    match = (online["ip_country"] == online["issuer_country"]).mean()
    assert match > 0.9


def naive_features(txns: pl.DataFrame) -> pl.DataFrame:
    """Recompute the point-in-time features the slow, obvious way, row by row."""
    rows = txns.sort(["account_id", "occurred_at"]).to_dicts()
    out = []
    for i, row in enumerate(rows):
        prior = [r for r in rows[:i] if r["account_id"] == row["account_id"]]
        t = row["occurred_at"]
        before = [r for r in prior if r["occurred_at"] < t]

        def within(r, window, t=t):
            return t - window <= r["occurred_at"] < t

        last30 = [r for r in before if within(r, timedelta(days=30))]
        total = sum(r["amount_minor"] for r in last30)
        same_card_merchant = [
            r
            for r in prior
            if (r["card_bin"], r["merchant_id"]) == (row["card_bin"], row["merchant_id"])
        ]
        out.append(
            {
                "transaction_id": row["transaction_id"],
                "txn_count_1h": sum(within(r, timedelta(hours=1)) for r in before),
                "txn_count_24h": sum(within(r, timedelta(hours=24)) for r in before),
                "avg_amount_30d_minor": (2 * total + len(last30)) // (2 * len(last30))
                if last30
                else None,
                "seconds_since_last_txn": int((t - prior[-1]["occurred_at"]).total_seconds())
                if prior
                else None,
                "device_is_new_for_account": row["channel"] != "pos"
                and all(r["device_id"] != row["device_id"] for r in prior),
                "card_is_new_for_merchant": not same_card_merchant,
            }
        )
    return pl.DataFrame(out, schema_overrides={"avg_amount_30d_minor": pl.Int64})


def test_features_match_a_naive_row_by_row_recomputation(tiny):
    _, _, txns = tiny
    columns = [
        "transaction_id",
        "txn_count_1h",
        "txn_count_24h",
        "avg_amount_30d_minor",
        "seconds_since_last_txn",
        "device_is_new_for_account",
        "card_is_new_for_merchant",
    ]
    expected = naive_features(txns).sort("transaction_id")
    actual = txns.select(columns).sort("transaction_id")
    assert actual.equals(expected.select(columns))


def test_features_never_look_at_the_future(tiny):
    """Dropping everything after a cutoff must not change any feature before it."""
    universe, _, txns = tiny
    raw = txns.join(
        universe.accounts.select("account_id", "created_at"), on="account_id", how="left"
    ).select(
        "transaction_id",
        "account_id",
        "occurred_at",
        "merchant_id",
        "amount_minor",
        "device_id",
        "card_bin",
        "created_at",
    )
    cutoff = universe.profile.window_start + timedelta(days=universe.profile.days // 2)
    features = ["txn_count_1h", "txn_count_24h", "avg_amount_30d_minor", "seconds_since_last_txn"]

    full = add_point_in_time_features(raw).sort("transaction_id")
    truncated = add_point_in_time_features(raw.filter(pl.col("occurred_at") < cutoff)).sort(
        "transaction_id"
    )
    before = full.filter(pl.col("occurred_at") < cutoff)

    assert truncated.height == before.height > 0
    assert truncated.select(features).equals(before.select(features))
