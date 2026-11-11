"""Decision bands: where a ranking score turns into APPROVE, REVIEW or DECLINE.

The thresholds are not hand-picked scores: they are chosen on the **validation** set so that a
target share of the legitimate (label-negative) transactions lands above them:

* ``review`` at a false-positive rate of 2% (the manual-review queue the business can staff),
* ``decline`` at 0.2% (blocking a legitimate customer is the most expensive mistake).

Bands: ``APPROVE`` if ``score < review``; ``REVIEW`` if ``review <= score < decline``;
else ``DECLINE``.
"""

from dataclasses import asdict, dataclass
from typing import Any, Self

import numpy as np

from risk_scoring.training.metrics import threshold_for_fpr

DEFAULT_REVIEW_FPR = 0.02
DEFAULT_DECLINE_FPR = 0.002
BANDS = ("APPROVE", "REVIEW", "DECLINE")


@dataclass(frozen=True)
class BandThresholds:
    review: float
    decline: float
    review_fpr: float = DEFAULT_REVIEW_FPR
    decline_fpr: float = DEFAULT_DECLINE_FPR

    def __post_init__(self) -> None:
        if self.review > self.decline:
            raise ValueError("the review threshold cannot exceed the decline threshold")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)


def choose_thresholds(
    y_validation: np.ndarray,
    scores_validation: np.ndarray,
    review_fpr: float = DEFAULT_REVIEW_FPR,
    decline_fpr: float = DEFAULT_DECLINE_FPR,
) -> BandThresholds:
    if decline_fpr > review_fpr:
        raise ValueError("decline_fpr must not exceed review_fpr")
    review = threshold_for_fpr(y_validation, scores_validation, review_fpr)
    decline = threshold_for_fpr(y_validation, scores_validation, decline_fpr)
    if decline > 1.0 or review > 1.0:
        raise ValueError("the false-positive budget cannot be met: no score is low enough")
    return BandThresholds(review, decline, review_fpr, decline_fpr)


def assign_bands(scores: np.ndarray, thresholds: BandThresholds) -> np.ndarray:
    """The band of each score, as an array of strings from ``BANDS``."""
    scores = np.asarray(scores)
    return np.where(
        scores >= thresholds.decline,
        "DECLINE",
        np.where(scores >= thresholds.review, "REVIEW", "APPROVE"),
    )
