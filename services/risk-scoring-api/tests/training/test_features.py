import math
from datetime import UTC, datetime

import polars as pl
import pytest

from payments_shared.data_contracts import Role, load_contract
from risk_scoring.training.features import (
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    FEATURE_GROUPS,
    NUMERIC_COLUMNS,
    derive_features,
    required_columns,
)


def raw_frame(*rows: dict) -> pl.DataFrame:
    contract = load_contract("transactions")
    schema = {name: contract.schema[name] for name in required_columns()}
    return pl.DataFrame(list(rows), schema=schema)


def base_row(**patch) -> dict:
    row = {
        "occurred_at": datetime(2026, 1, 12, 22, 41, 7, tzinfo=UTC),  # a Monday
        "amount_minor": 10_000,
        "mcc": "5732",
        "channel": "web",
        "installments": 3,
        "card_bin": "912345",
        "issuer_country": "BR",
        "ip_country": "RU",
        "is_proxy": True,
        "device_is_new_for_account": True,
        "card_is_new_for_merchant": False,
        "account_age_days": 9,
        "txn_count_1h": 2,
        "txn_count_24h": 5,
        "avg_amount_30d_minor": 5_000,
        "seconds_since_last_txn": 240,
    }
    return {**row, **patch}


def derive_one(**patch) -> dict:
    return derive_features(raw_frame(base_row(**patch))).row(0, named=True)


def test_required_columns_are_exactly_the_contract_feature_columns():
    contract = load_contract("transactions")
    assert set(required_columns()) == {c.name for c in contract.columns_with_role(Role.FEATURE)}


def test_no_label_or_analysis_column_can_reach_the_pipeline():
    contract = load_contract("transactions")
    forbidden = {
        c.name
        for role in (Role.ANALYSIS_ONLY, Role.LABEL_SOURCE, Role.ID)
        for c in contract.columns_with_role(role)
    }
    derived_inputs = set(required_columns()) | set(NUMERIC_COLUMNS + BINARY_COLUMNS)
    assert not forbidden & derived_inputs
    assert not forbidden & set(derive_features(raw_frame(base_row())).columns)


def test_golden_values_of_a_risky_web_transaction():
    row = derive_one()

    assert row["log_amount"] == pytest.approx(math.log1p(10_000))
    assert row["log_account_age_days"] == pytest.approx(math.log1p(9))
    assert row["log_txn_count_1h"] == pytest.approx(math.log1p(2))
    assert row["log_txn_count_24h"] == pytest.approx(math.log1p(5))
    assert row["log_avg_amount_30d"] == pytest.approx(math.log1p(5_000))
    assert row["log_seconds_since_last_txn"] == pytest.approx(math.log1p(240))
    assert row["amount_to_avg_ratio"] == pytest.approx(2.0)
    assert row["installments"] == 3.0
    # 22h on the 24h circle; Monday is day 0 of 7.
    assert row["hour_sin"] == pytest.approx(math.sin(2 * math.pi * 22 / 24))
    assert row["hour_cos"] == pytest.approx(math.cos(2 * math.pi * 22 / 24))
    assert row["dow_sin"] == pytest.approx(0.0, abs=1e-12)
    assert row["dow_cos"] == pytest.approx(1.0)
    assert [row[c] for c in BINARY_COLUMNS] == [
        0.0,  # avg_amount_30d_missing
        0.0,  # seconds_since_last_txn_missing
        1.0,  # country_mismatch (RU vs BR)
        1.0,  # ip_foreign
        0.0,  # issuer_foreign
        0.0,  # ip_country_missing
        1.0,  # is_proxy
        1.0,  # device_is_new_for_account
        0.0,  # card_is_new_for_merchant
    ]
    assert [row[c] for c in CATEGORICAL_COLUMNS] == ["5732", "912345", "web", "BR", "RU"]


def test_first_in_person_transaction_has_missing_indicators_and_neutral_fills():
    row = derive_one(
        occurred_at=datetime(2026, 1, 17, 0, 0, 0, tzinfo=UTC),  # a Saturday, midnight
        channel="pos",
        ip_country=None,
        is_proxy=False,
        device_is_new_for_account=False,
        issuer_country="AR",
        txn_count_1h=0,
        txn_count_24h=0,
        avg_amount_30d_minor=None,
        seconds_since_last_txn=None,
        amount_minor=50,
    )

    assert row["avg_amount_30d_missing"] == 1.0
    assert row["seconds_since_last_txn_missing"] == 1.0
    assert row["log_avg_amount_30d"] == 0.0
    assert row["log_seconds_since_last_txn"] == 0.0
    assert row["amount_to_avg_ratio"] == 1.0
    assert row["ip_country_missing"] == 1.0
    assert row["country_mismatch"] == 0.0  # unknown is not a mismatch
    assert row["ip_foreign"] == 0.0
    assert row["issuer_foreign"] == 1.0
    assert row["ip_country"] is None
    assert row["hour_sin"] == pytest.approx(0.0, abs=1e-12) and row["hour_cos"] == 1.0
    # Saturday is day 5 of 7.
    assert row["dow_sin"] == pytest.approx(math.sin(2 * math.pi * 5 / 7))
    assert row["dow_cos"] == pytest.approx(math.cos(2 * math.pi * 5 / 7))


@pytest.mark.parametrize(
    "amount, avg, expected",
    [(5_000_000, 100, 20.0), (50, 5_000, 0.01), (7_500, 5_000, 1.5), (100, 0, 1.0)],
)
def test_amount_to_avg_ratio_is_clipped_and_safe(amount, avg, expected):
    assert derive_one(amount_minor=amount, avg_amount_30d_minor=avg)[
        "amount_to_avg_ratio"
    ] == pytest.approx(expected)


def test_time_features_use_utc_whatever_the_input_offset():
    from datetime import timedelta, timezone

    sao_paulo = timezone(timedelta(hours=-3))
    local = datetime(2026, 1, 12, 19, 41, 7, tzinfo=sao_paulo)  # same instant as 22:41:07 UTC
    assert derive_one(occurred_at=local)["hour_sin"] == pytest.approx(derive_one()["hour_sin"])


def test_missing_input_column_is_reported_by_name():
    with pytest.raises(ValueError, match="amount_minor"):
        derive_features(raw_frame(base_row()).drop("amount_minor"))


def test_extra_columns_are_ignored_not_leaked(sample_transactions):
    derived = derive_features(sample_transactions)  # still has is_fraud, ids, labels...
    assert set(derived.columns) == set(NUMERIC_COLUMNS + BINARY_COLUMNS + CATEGORICAL_COLUMNS)
    assert derived.height == sample_transactions.height


def test_single_row_path_equals_batch_path(sample_transactions):
    batch = derive_features(sample_transactions.head(200))
    for i in (0, 1, 57, 199):
        single = derive_features(sample_transactions.slice(i, 1))
        for column in (*NUMERIC_COLUMNS, *BINARY_COLUMNS):
            assert single[column][0] == pytest.approx(batch[column][i], abs=1e-12), column
        for column in CATEGORICAL_COLUMNS:
            assert single[column][0] == batch[column][i], column


def test_feature_groups_cover_every_derived_column_exactly_once():
    grouped = [c for columns in FEATURE_GROUPS.values() for c in columns]
    everything = [*NUMERIC_COLUMNS, *BINARY_COLUMNS, *CATEGORICAL_COLUMNS]
    assert sorted(grouped) == sorted(everything)


def test_derived_features_have_no_nulls_except_ip_country(sample_transactions):
    derived = derive_features(sample_transactions)
    nulls = {c: derived[c].null_count() for c in derived.columns if derived[c].null_count()}
    assert set(nulls) <= {"ip_country"}
