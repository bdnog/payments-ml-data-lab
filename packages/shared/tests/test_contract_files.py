"""Every committed YAML data contract must load and be internally consistent."""

import polars as pl
import pytest

from payments_shared.data_contracts import Role, list_contracts, load_contract, validate_dataframe

EXPECTED = {
    "merchants",
    "accounts",
    "transactions",
    "payments",
    "refunds",
    "chargebacks",
    "psp_settlement",
    "injected_breaks",
    "daily_settlement_by_merchant",
    "reconciliation_breaks",
    "fee_summary",
}


def test_all_expected_contracts_exist():
    assert set(list_contracts()) == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_contract_loads_and_is_consistent(name):
    contract = load_contract(name)

    assert contract.version >= 1
    assert contract.description
    assert all(col.description for col in contract.columns), "every column needs a description"
    assert set(contract.primary_key) <= set(contract.column_names)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_rules_compile_against_the_contract_schema(name):
    """An empty frame with the right schema must validate: proves each rule's SQL is well formed."""
    contract = load_contract(name)
    empty = pl.DataFrame(schema=contract.schema)

    assert validate_dataframe(empty, contract) == []


def test_transactions_roles_follow_the_label_policy():
    contract = load_contract("transactions")

    assert {c.name for c in contract.columns_with_role(Role.ANALYSIS_ONLY)} == {
        "is_fraud",
        "fraud_pattern",
    }
    assert [c.name for c in contract.columns_with_role(Role.LABEL_SOURCE)] == [
        "chargeback_reported_at"
    ]
    assert {c.name for c in contract.columns_with_role(Role.FEATURE)} == {
        "amount_minor",
        "mcc",
        "channel",
        "installments",
        "card_bin",
        "issuer_country",
        "ip_country",
        "is_proxy",
        "device_is_new_for_account",
        "card_is_new_for_merchant",
        "account_age_days",
        "txn_count_1h",
        "txn_count_24h",
        "avg_amount_30d_minor",
        "seconds_since_last_txn",
    }
