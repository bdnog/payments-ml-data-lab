"""Ranking metrics in plain NumPy (scikit-learn is a dev dependency used only to test these).

Conventions, all matching scikit-learn:

* ``y_true`` is a 0/1 (or bool) array, ``scores`` a float array; higher means more likely fraud.
* Ties are handled by grouping equal scores: a threshold either includes all of a tie group or
  none of it. Without this, the order of equal scores would change the metric.
* Metrics that are undefined (no positives, or no negatives) return ``nan`` instead of raising, so
  a report on a tiny split still renders.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Counts:
    """Confusion counts at every distinct score threshold ("flag score >= threshold")."""

    thresholds: np.ndarray  # descending, distinct
    tp: np.ndarray
    fp: np.ndarray
    n_pos: int
    n_neg: int


def _prepare(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError(
            f"y_true and scores must be 1-D arrays of the same length: {y.shape}, {s.shape}"
        )
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    return y, s


def counts_at_thresholds(y_true: np.ndarray, scores: np.ndarray) -> Counts:
    y, s = _prepare(y_true, scores)
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    last_of_group = np.append(ss[1:] != ss[:-1], True) if len(ss) else np.array([], dtype=bool)
    tp = np.cumsum(ys)[last_of_group]
    fp = np.cumsum(~ys)[last_of_group]
    return Counts(ss[last_of_group], tp, fp, int(y.sum()), int((~y).sum()))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """PR-AUC as average precision: sum over thresholds of (recall step) * precision (step-wise)."""
    c = counts_at_thresholds(y_true, scores)
    if c.n_pos == 0:
        return float("nan")
    precision = c.tp / (c.tp + c.fp)
    recall = c.tp / c.n_pos
    return float(np.sum(np.diff(recall, prepend=0.0) * precision))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    c = counts_at_thresholds(y_true, scores)
    if c.n_pos == 0 or c.n_neg == 0:
        return float("nan")
    tpr = np.concatenate([[0.0], c.tp / c.n_pos])
    fpr = np.concatenate([[0.0], c.fp / c.n_neg])
    return float(np.trapezoid(tpr, fpr))


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr: float) -> float:
    """Best recall reachable while flagging at most ``fpr`` of the negatives."""
    c = counts_at_thresholds(y_true, scores)
    if c.n_pos == 0 or c.n_neg == 0:
        return float("nan")
    allowed = c.fp / c.n_neg <= fpr
    return float(c.tp[allowed].max() / c.n_pos) if allowed.any() else 0.0


def threshold_for_fpr(y_true: np.ndarray, scores: np.ndarray, fpr: float) -> float:
    """Lowest threshold such that ``score >= threshold`` flags at most ``fpr`` of the negatives.

    If even the highest score would exceed the budget, returns a value above every score (nothing
    is flagged). Ties make the realized FPR lower than requested, never higher.
    """
    if not 0 <= fpr <= 1:
        raise ValueError("fpr must be in [0, 1]")
    c = counts_at_thresholds(y_true, scores)
    if c.n_neg == 0:
        raise ValueError("cannot choose a threshold without negatives")
    allowed = np.flatnonzero(c.fp / c.n_neg <= fpr)
    if allowed.size == 0:
        return float(np.nextafter(c.thresholds[0], np.inf))
    return float(c.thresholds[allowed.max()])


def amount_weighted_recall(y_true: np.ndarray, flagged: np.ndarray, amounts: np.ndarray) -> float:
    """Share of the money at risk that was caught: flagged fraud amount over total fraud amount."""
    y = np.asarray(y_true).astype(bool)
    caught = np.asarray(flagged).astype(bool)
    money = np.asarray(amounts, dtype=np.float64)
    total = money[y].sum()
    if total == 0:
        return float("nan")
    return float(money[y & caught].sum() / total)


def brier_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mean squared error of the score read as a probability (our scores are not calibrated)."""
    y, s = _prepare(y_true, scores)
    return float(np.mean((s - y) ** 2))


def precision_recall_at(y_true: np.ndarray, flagged: np.ndarray) -> tuple[float, float]:
    """(precision, recall) of a flagged set; ``nan`` where undefined."""
    y = np.asarray(y_true).astype(bool)
    caught = np.asarray(flagged).astype(bool)
    tp = float((y & caught).sum())
    precision = tp / caught.sum() if caught.any() else float("nan")
    recall = tp / y.sum() if y.any() else float("nan")
    return precision, recall


def score_deciles(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Rows sorted by score, cut in ``n_bins`` equal-size groups; bin 1 holds the highest scores."""
    y, s = _prepare(y_true, scores)
    order = np.argsort(-s, kind="stable")
    rows = []
    for number, index in enumerate(np.array_split(order, n_bins), start=1):
        if index.size == 0:
            continue
        rows.append(
            {
                "decile": number,
                "rows": int(index.size),
                "positives": int(y[index].sum()),
                "positive_rate": float(y[index].mean()),
                "min_score": float(s[index].min()),
                "max_score": float(s[index].max()),
            }
        )
    return rows


def prevalence(y_true: np.ndarray) -> float:
    """Share of positives: the PR-AUC of a random classifier, the reference line for PR-AUC."""
    y = np.asarray(y_true).astype(bool)
    return float(y.mean()) if y.size else float("nan")
