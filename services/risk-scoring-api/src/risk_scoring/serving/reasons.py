"""Reason codes: heuristic explanations of a high score.

They are rules over the request values, **not attributions of the model** (the model is a neural
network; explaining it faithfully would need SHAP or integrated gradients, a future extension).
This module holds the rule thresholds; the rules themselves are added with the scorer.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Self

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
