import polars as pl
import pytest

from payments_datagen.catalog import MCC_TABLE
from payments_datagen.profiles import FULL, SAMPLE, TINY, Profile, get_profile
from payments_datagen.universe import build_universe
from payments_shared.data_contracts import assert_valid, check_foreign_key, load_contract


@pytest.fixture(scope="module")
def universe():
    return build_universe(TINY)


def test_profiles_match_the_spec():
    assert (SAMPLE.n_accounts, SAMPLE.n_merchants, SAMPLE.days) == (300, 25, 30)
    assert (FULL.n_accounts, FULL.n_merchants, FULL.days) == (5_000, 200, 180)
    assert (SAMPLE.fraud_rate, SAMPLE.break_rate) == (0.03, 0.06)
    assert (FULL.fraud_rate, FULL.break_rate) == (0.01, 0.02)
    assert SAMPLE.seed == 42
    assert SAMPLE.start_date.isoformat() == "2026-01-01"


def test_overrides_replace_only_given_values():
    profile = get_profile("sample", seed=7, days=10, fraud_rate=None)

    assert (profile.seed, profile.days, profile.fraud_rate) == (7, 10, 0.03)
    assert profile.last_day.isoformat() == "2026-01-10"


@pytest.mark.parametrize("bad", [{"days": 0}, {"fraud_rate": 0.9}, {"break_rate": -0.1}])
def test_invalid_profile_values_are_rejected(bad):
    with pytest.raises(ValueError):
        get_profile("sample", **bad)


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown profile"):
        get_profile("huge")


def test_reference_datasets_honor_their_contracts(universe):
    assert_valid(universe.merchants, load_contract("merchants"))
    assert_valid(universe.accounts, load_contract("accounts"))


def test_ids_grow_with_creation_time(universe):
    for frame, key in ((universe.merchants, "merchant_id"), (universe.accounts, "account_id")):
        assert frame[key].is_sorted()
        assert frame["created_at"].is_sorted()


def test_sizes_follow_the_profile(universe):
    assert universe.merchants.height == TINY.n_merchants
    assert universe.accounts.height == TINY.n_accounts
    assert universe.traits.height == TINY.n_accounts


def test_every_mcc_is_present_when_there_are_enough_merchants(universe):
    assert set(universe.merchants["mcc"]) == set(MCC_TABLE["mcc"])


def test_some_accounts_predate_the_window_and_some_are_new(universe):
    start = TINY.window_start
    created = universe.accounts["created_at"]
    assert (created < start).any()
    assert (created >= start).any()


def test_latent_tables_reference_real_entities(universe):
    assert check_foreign_key(universe.devices, "account_id", universe.accounts, "account_id") == []
    assert (
        check_foreign_key(universe.terminals, "merchant_id", universe.merchants, "merchant_id")
        == []
    )
    assert check_foreign_key(universe.traits, "card_bin", universe.bins, "card_bin") == []
    all_devices = pl.concat([universe.devices["device_id"], universe.terminals["device_id"]])
    assert all_devices.n_unique() == len(all_devices)


def test_bin_pool_is_forty_six_digit_synthetic_bins(universe):
    assert universe.bins.height == 40
    assert universe.bins["card_bin"].n_unique() == 40
    assert universe.bins["card_bin"].str.contains(r"^9[0-9]{5}$").all()


def test_merchant_names_are_generic_and_unique(universe):
    assert universe.merchants["name"].n_unique() == TINY.n_merchants
    assert universe.merchants["name"].str.contains(r"^[A-Z][a-z]+ [A-Z][a-z]+ [0-9]{4}$").all()


def test_same_seed_gives_equal_universe_and_different_seed_differs():
    again = build_universe(TINY)
    other = build_universe(Profile(**{**TINY.__dict__, "seed": 43}))
    first = build_universe(TINY)

    assert first.merchants.equals(again.merchants)
    assert first.accounts.equals(again.accounts)
    assert first.traits.equals(again.traits)
    assert not first.accounts.equals(other.accounts)
