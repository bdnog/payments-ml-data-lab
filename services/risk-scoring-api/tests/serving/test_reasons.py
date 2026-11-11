import json

import pytest

from payments_shared.paths import contracts_dir
from risk_scoring.api.schemas import ReasonCode, ScoreRequest
from risk_scoring.serving.reasons import (
    REASON_PRIORITY,
    ReasonRules,
    fired_rules,
    reason_codes,
)

EXAMPLE_FILE = contracts_dir() / "examples" / "risk-scoring" / "score.request.json"
EXAMPLE = json.loads(EXAMPLE_FILE.read_text())


def request(**patch) -> ScoreRequest:
    """The contract example, with dotted paths patched: ``request(**{"device.is_proxy": 0})``."""
    body = json.loads(json.dumps(EXAMPLE))
    for path, value in patch.items():
        *parents, leaf = path.split(".")
        target = body
        for parent in parents:
            target = target[parent]
        target[leaf] = value
    return ScoreRequest.model_validate(body)


def calm(**patch) -> ScoreRequest:
    """A request that triggers no rule at all, then patched."""
    base = {
        "device.is_proxy": False,
        "device.is_new_for_account": False,
        "device.ip_country": "BR",
        "card.issuer_country": "BR",
        "card.is_new_for_merchant": False,
        "amount_minor": 1_000,
        "behavior.avg_amount_30d_minor": 1_000,
        "behavior.txn_count_1h": 0,
        "behavior.txn_count_24h": 0,
        "account.age_days": 400,
        "merchant.mcc": "5411",
    }
    return request(**{**base, **patch})


RULES = ReasonRules()


def test_a_calm_request_fires_nothing():
    assert fired_rules(calm(), RULES) == set()


@pytest.mark.parametrize(
    "patch, code",
    [
        ({"device.is_proxy": True}, ReasonCode.PROXY_DETECTED),
        ({"device.is_new_for_account": True}, ReasonCode.NEW_DEVICE),
        ({"device.ip_country": "RU"}, ReasonCode.COUNTRY_MISMATCH),
        ({"amount_minor": 3_000}, ReasonCode.AMOUNT_ABOVE_NORMAL),
        ({"behavior.txn_count_1h": 5, "behavior.txn_count_24h": 5}, ReasonCode.HIGH_VELOCITY_1H),
        ({"behavior.txn_count_24h": 15}, ReasonCode.HIGH_VELOCITY_24H),
        ({"account.age_days": 29}, ReasonCode.NEW_ACCOUNT),
        ({"merchant.mcc": "5732"}, ReasonCode.HIGH_RISK_MCC),
        ({"card.is_new_for_merchant": True}, ReasonCode.NEW_CARD_FOR_MERCHANT),
    ],
)
def test_each_rule_fires_on_its_own(patch, code):
    assert fired_rules(calm(**patch), RULES) == {code}


@pytest.mark.parametrize(
    "patch",
    [
        {"amount_minor": 2_999},  # just below 3x the average
        {"behavior.txn_count_1h": 4, "behavior.txn_count_24h": 4},
        {"behavior.txn_count_24h": 14},
        {"account.age_days": 30},
        {"merchant.mcc": "5411"},
    ],
)
def test_rules_do_not_fire_just_below_their_thresholds(patch):
    assert fired_rules(calm(**patch), RULES) == set()


def test_missing_values_never_fire_a_rule():
    request_ = calm(
        **{
            "device.ip_country": None,
            "behavior.avg_amount_30d_minor": None,
            "amount_minor": 5_000_000,
        }
    )
    assert fired_rules(request_, RULES) == set()


def test_in_person_purchase_has_no_country_mismatch():
    assert ReasonCode.COUNTRY_MISMATCH not in fired_rules(
        calm(**{"device.ip_country": None}), RULES
    )


def test_approved_transactions_get_no_reasons_even_if_rules_fire():
    risky = calm(**{"device.is_proxy": True, "device.is_new_for_account": True})
    assert reason_codes(risky, RULES, flagged=False) == []
    assert reason_codes(risky, RULES, flagged=True) == [
        ReasonCode.PROXY_DETECTED,
        ReasonCode.NEW_DEVICE,
    ]


def test_reasons_follow_the_fixed_priority_and_are_capped():
    everything = calm(
        **{
            "device.is_proxy": True,
            "device.is_new_for_account": True,
            "device.ip_country": "RU",
            "amount_minor": 9_000,
            "behavior.txn_count_1h": 6,
            "behavior.txn_count_24h": 20,
            "account.age_days": 2,
            "merchant.mcc": "5732",
            "card.is_new_for_merchant": True,
        }
    )
    assert fired_rules(everything, RULES) == set(REASON_PRIORITY)

    codes = reason_codes(everything, RULES, flagged=True)

    assert codes == list(REASON_PRIORITY[:5])
    assert len(reason_codes(everything, ReasonRules(max_reasons=2), flagged=True)) == 2


def test_priority_covers_every_contract_reason_code_once():
    assert sorted(REASON_PRIORITY) == sorted(ReasonCode)


def test_thresholds_are_configurable():
    strict = ReasonRules(new_account_days=100, high_risk_mccs=("5411",))
    assert fired_rules(calm(**{"account.age_days": 99, "merchant.mcc": "5411"}), strict) == {
        ReasonCode.NEW_ACCOUNT,
        ReasonCode.HIGH_RISK_MCC,
    }
