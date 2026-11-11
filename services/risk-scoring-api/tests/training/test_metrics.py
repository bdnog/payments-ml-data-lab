"""The metrics are hand-written, so they are checked against scikit-learn (dev-only dependency)."""

import numpy as np
import pytest
from sklearn import metrics as sk

from risk_scoring.training import metrics as m

TOL = 1e-9


def random_case(seed: int, n: int = 2_000, positive_rate: float = 0.05, decimals: int | None = 1):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < positive_rate
    scores = np.clip(rng.normal(0.3 + 0.35 * y, 0.25), 0, 1)
    if decimals is not None:
        scores = scores.round(decimals)  # lots of ties on purpose
    return y, scores


CASES = [
    pytest.param(seed, decimals, id=f"seed{seed}-{'ties' if decimals else 'continuous'}")
    for seed in (1, 2, 3)
    for decimals in (1, 2, None)
]


@pytest.mark.parametrize("seed, decimals", CASES)
def test_average_precision_matches_sklearn(seed, decimals):
    y, s = random_case(seed, decimals=decimals)
    assert m.average_precision(y, s) == pytest.approx(sk.average_precision_score(y, s), abs=TOL)


@pytest.mark.parametrize("seed, decimals", CASES)
def test_roc_auc_matches_sklearn(seed, decimals):
    y, s = random_case(seed, decimals=decimals)
    assert m.roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s), abs=TOL)


@pytest.mark.parametrize("seed, decimals", CASES)
@pytest.mark.parametrize("fpr", [0.001, 0.01, 0.05, 0.3])
def test_recall_at_fpr_matches_the_sklearn_roc_curve(seed, decimals, fpr):
    y, s = random_case(seed, decimals=decimals)
    fprs, tprs, _ = sk.roc_curve(y, s, drop_intermediate=False)
    expected = tprs[fprs <= fpr].max()
    assert m.recall_at_fpr(y, s, fpr) == pytest.approx(expected, abs=TOL)


@pytest.mark.parametrize("seed, decimals", CASES)
def test_brier_score_matches_sklearn(seed, decimals):
    y, s = random_case(seed, decimals=decimals)
    assert m.brier_score(y, s) == pytest.approx(sk.brier_score_loss(y, s), abs=TOL)


def test_ties_do_not_depend_on_row_order():
    y, s = random_case(5, decimals=1)
    perm = np.random.default_rng(0).permutation(len(y))
    assert m.average_precision(y[perm], s[perm]) == m.average_precision(y, s)
    assert m.roc_auc(y[perm], s[perm]) == pytest.approx(m.roc_auc(y, s), abs=TOL)


def test_constant_scores_give_prevalence_and_half():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    s = np.full(10, 0.4)
    assert m.average_precision(y, s) == pytest.approx(0.2)
    assert m.roc_auc(y, s) == pytest.approx(0.5)


def test_perfect_and_inverted_rankings():
    y = np.array([1, 1, 0, 0, 0])
    assert m.average_precision(y, np.array([0.9, 0.8, 0.3, 0.2, 0.1])) == 1.0
    assert m.roc_auc(y, np.array([0.9, 0.8, 0.3, 0.2, 0.1])) == 1.0
    assert m.roc_auc(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9])) == 0.0


def test_undefined_metrics_are_nan_not_errors():
    s = np.array([0.1, 0.2, 0.3])
    assert np.isnan(m.average_precision(np.zeros(3), s))
    assert np.isnan(m.roc_auc(np.zeros(3), s)) and np.isnan(m.roc_auc(np.ones(3), s))
    assert np.isnan(m.recall_at_fpr(np.zeros(3), s, 0.01))
    assert np.isnan(m.amount_weighted_recall(np.zeros(3), np.ones(3), np.ones(3)))


def test_amount_weighted_recall_on_a_hand_computed_example():
    # Fraud amounts: 100 (caught) + 300 (missed) + 600 (caught) = 1000; the legit 5000 is ignored.
    y = np.array([1, 1, 1, 0, 0])
    flagged = np.array([1, 0, 1, 1, 0])
    amounts = np.array([100, 300, 600, 5_000, 70])
    assert m.amount_weighted_recall(y, flagged, amounts) == pytest.approx(0.7)
    # By count the recall is 2/3, by money 70%: the metric is not the plain recall.
    assert m.precision_recall_at(y, flagged)[1] == pytest.approx(2 / 3)


def test_precision_and_recall_of_a_flagged_set():
    y = np.array([1, 1, 0, 0, 0, 0])
    flagged = np.array([1, 0, 1, 1, 0, 0])
    precision, recall = m.precision_recall_at(y, flagged)
    assert (precision, recall) == (pytest.approx(1 / 3), pytest.approx(0.5))
    assert np.isnan(m.precision_recall_at(y, np.zeros(6))[0])


def test_threshold_for_fpr_respects_the_budget_and_ties():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 20)
    s = np.tile([0.9, 0.5, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1, 0.1, 0.0], 20)
    n_neg = (~y.astype(bool)).sum()

    for fpr in (0.0, 0.05, 0.1, 0.2, 0.5):
        t = m.threshold_for_fpr(y, s, fpr)
        flagged_negatives = ((s >= t) & ~y.astype(bool)).sum()
        assert flagged_negatives / n_neg <= fpr
    # A budget of 0 flags only what scores above every negative: here the single 0.9 group.
    assert m.threshold_for_fpr(y, s, 0.0) == 0.9
    # The recall at the chosen threshold equals recall_at_fpr.
    t = m.threshold_for_fpr(y, s, 0.1)
    assert (s[y.astype(bool)] >= t).mean() == m.recall_at_fpr(y, s, 0.1)


def test_threshold_above_every_score_when_nothing_fits_the_budget():
    y = np.array([0, 0, 1])
    s = np.array([0.9, 0.8, 0.1])
    t = m.threshold_for_fpr(y, s, 0.1)
    assert t > s.max() and not (s >= t).any()


def test_score_deciles_partition_the_rows_and_concentrate_positives_at_the_top():
    y, s = random_case(9, n=1_000, decimals=None)
    deciles = m.score_deciles(y, s)

    assert [d["decile"] for d in deciles] == list(range(1, 11))
    assert sum(d["rows"] for d in deciles) == 1_000
    assert sum(d["positives"] for d in deciles) == y.sum()
    assert deciles[0]["positive_rate"] > deciles[-1]["positive_rate"]
    assert all(a["min_score"] >= b["max_score"] for a, b in zip(deciles, deciles[1:], strict=False))


def test_prevalence_is_the_share_of_positives():
    assert m.prevalence(np.array([1, 0, 0, 0])) == 0.25


def test_bad_input_shapes_and_non_finite_scores_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        m.average_precision(np.array([1, 0]), np.array([0.1]))
    with pytest.raises(ValueError, match="finite"):
        m.roc_auc(np.array([1, 0]), np.array([0.1, np.nan]))
