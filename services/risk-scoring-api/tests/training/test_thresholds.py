import numpy as np
import pytest

from risk_scoring.training.thresholds import BandThresholds, assign_bands, choose_thresholds


def validation_case(seed: int = 0, n: int = 20_000):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.03
    scores = np.clip(rng.normal(0.2 + 0.5 * y, 0.2), 0, 1)
    return y, scores


def test_thresholds_respect_the_false_positive_budgets():
    y, scores = validation_case()
    t = choose_thresholds(y, scores, review_fpr=0.02, decline_fpr=0.002)

    negatives = scores[~y]
    assert (negatives >= t.review).mean() <= 0.02
    assert (negatives >= t.decline).mean() <= 0.002
    # ...and they are not wastefully conservative: close to the budget on continuous scores.
    assert (negatives >= t.review).mean() > 0.015
    assert t.review < t.decline


def test_stricter_budget_gives_a_higher_threshold():
    y, scores = validation_case(1)
    loose = choose_thresholds(y, scores, 0.05, 0.01)
    strict = choose_thresholds(y, scores, 0.02, 0.002)
    assert strict.review > loose.review and strict.decline > loose.decline


def test_decline_budget_cannot_exceed_review_budget():
    y, scores = validation_case()
    with pytest.raises(ValueError, match="decline_fpr"):
        choose_thresholds(y, scores, review_fpr=0.01, decline_fpr=0.02)


def test_review_cannot_exceed_decline():
    with pytest.raises(ValueError, match="review"):
        BandThresholds(review=0.9, decline=0.5)


def test_bands_follow_the_documented_boundaries():
    t = BandThresholds(review=0.5, decline=0.85)
    scores = np.array([0.0, 0.4999, 0.5, 0.8499, 0.85, 1.0])
    assert assign_bands(scores, t).tolist() == [
        "APPROVE",
        "APPROVE",
        "REVIEW",
        "REVIEW",
        "DECLINE",
        "DECLINE",
    ]


def test_equal_thresholds_leave_no_review_band():
    t = BandThresholds(review=0.7, decline=0.7)
    assert set(assign_bands(np.array([0.1, 0.7, 0.9]), t)) == {"APPROVE", "DECLINE"}


def test_thresholds_round_trip_through_a_dict():
    t = BandThresholds(review=0.3, decline=0.6, review_fpr=0.03, decline_fpr=0.003)
    assert BandThresholds.from_dict(t.to_dict()) == t
