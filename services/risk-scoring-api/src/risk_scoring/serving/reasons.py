"""Reason codes: heuristic explanations of a high score.

They are rules over the request values, **not attributions of the model** (the model is a neural
network; explaining it faithfully would need SHAP or integrated gradients, a future extension).
The rules are applied only to transactions that are not approved (REVIEW and DECLINE): the
contract says the list is "empty for clearly legitimate transactions", and a reason is an answer to
"why was this flagged?". At most ``max_reasons`` codes are returned, in the fixed priority order of
``REASON_PRIORITY``.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Self

from risk_scoring.api.schemas import ReasonCode, ScoreRequest

# Categories the synthetic generator treats as high risk: electronics, jewelers, games, direct
# marketing and money transfer. Configurable on purpose: in a real system risk owns this list.
DEFAULT_HIGH_RISK_MCCS = ("4829", "5732", "5816", "5944", "5967")


@dataclass(frozen=True)
class ReasonRules:
    """The documented thresholds of the reason rules. Stored in the artifact metadata."""

    amount_above_normal_ratio: float = 3.0  # amount >= 3x the 30-day average
    high_velocity_1h: int = 5  # 5 or more transactions in the last hour
    high_velocity_24h: int = 15  # 15 or more in the last 24 hours
    new_account_days: int = 30  # account younger than 30 days
    high_risk_mccs: tuple[str, ...] = field(default=DEFAULT_HIGH_RISK_MCCS)
    max_reasons: int = 5

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["high_risk_mccs"] = list(self.high_risk_mccs)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**{**data, "high_risk_mccs": tuple(data["high_risk_mccs"])})


# Strongest signals first; the list is cut at ``max_reasons`` in this order.
REASON_PRIORITY: tuple[ReasonCode, ...] = (
    ReasonCode.PROXY_DETECTED,
    ReasonCode.NEW_DEVICE,
    ReasonCode.COUNTRY_MISMATCH,
    ReasonCode.AMOUNT_ABOVE_NORMAL,
    ReasonCode.HIGH_VELOCITY_1H,
    ReasonCode.HIGH_VELOCITY_24H,
    ReasonCode.NEW_ACCOUNT,
    ReasonCode.HIGH_RISK_MCC,
    ReasonCode.NEW_CARD_FOR_MERCHANT,
)


def fired_rules(request: ScoreRequest, rules: ReasonRules) -> set[ReasonCode]:
    """Every rule that fires for the request, in no particular order."""
    avg = request.behavior.avg_amount_30d_minor
    ip_country = request.device.ip_country
    checks = {
        ReasonCode.PROXY_DETECTED: request.device.is_proxy,
        ReasonCode.NEW_DEVICE: request.device.is_new_for_account,
        ReasonCode.COUNTRY_MISMATCH: (
            ip_country is not None and ip_country != request.card.issuer_country
        ),
        ReasonCode.AMOUNT_ABOVE_NORMAL: (
            avg is not None
            and avg > 0
            and request.amount_minor >= rules.amount_above_normal_ratio * avg
        ),
        ReasonCode.HIGH_VELOCITY_1H: request.behavior.txn_count_1h >= rules.high_velocity_1h,
        ReasonCode.HIGH_VELOCITY_24H: request.behavior.txn_count_24h >= rules.high_velocity_24h,
        ReasonCode.NEW_ACCOUNT: request.account.age_days < rules.new_account_days,
        ReasonCode.HIGH_RISK_MCC: request.merchant.mcc in rules.high_risk_mccs,
        ReasonCode.NEW_CARD_FOR_MERCHANT: request.card.is_new_for_merchant,
    }
    return {code for code, fired in checks.items() if fired}


def reason_codes(request: ScoreRequest, rules: ReasonRules, *, flagged: bool) -> list[ReasonCode]:
    """Reason codes of a decision: none if approved, else at most N, in priority order."""
    if not flagged:
        return []
    fired = fired_rules(request, rules)
    return [code for code in REASON_PRIORITY if code in fired][: rules.max_reasons]
