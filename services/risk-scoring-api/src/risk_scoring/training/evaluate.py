"""Evaluation of a trained model: per split, per band, per fraud pattern, per feature group.

Everything here is *reporting*. ``is_fraud`` and ``fraud_pattern`` (``analysis_only`` columns) are
used to ask "how much real fraud do we catch, and of which kind", never to train or pick weights.
The result is a plain nested dict of numbers, so it serializes to ``metrics.json`` unchanged.
"""

import math
from typing import Any

import numpy as np

from risk_scoring.training import metrics as m
from risk_scoring.training.dataset import PreparedData, SplitData
from risk_scoring.training.features import CATEGORICAL_COLUMNS, FEATURE_GROUPS
from risk_scoring.training.model import FraudMLP
from risk_scoring.training.preprocessing import Encoded
from risk_scoring.training.thresholds import BANDS, BandThresholds, assign_bands
from risk_scoring.training.trainer import predict_scores

FRAUD_PATTERNS = ("CARD_TESTING", "ACCOUNT_TAKEOVER", "GEO_MISMATCH", "VELOCITY_BURST")
SANITY_MIN_LIFT = 3.0
SANITY_SUSPICIOUS_PR_AUC = 0.98


def evaluate(
    data: PreparedData,
    model: FraudMLP,
    thresholds: BandThresholds,
    *,
    permutation_repeats: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    scores = {name: predict_scores(model, s.encoded) for name, s in data.splits.items()}
    held_out = ("validation", "test")
    result: dict[str, Any] = {
        "splits": {
            name: split_metrics(split, scores[name], thresholds)
            for name, split in data.splits.items()
        },
        "bands": {n: band_table(data.splits[n], scores[n], thresholds) for n in held_out},
        "patterns": {n: pattern_recall(data.splits[n], scores[n], thresholds) for n in held_out},
        "label_coverage": {n: label_coverage(s) for n, s in data.splits.items()},
        "permutation_importance": permutation_importance(
            data, model, data.splits["validation"], permutation_repeats, seed
        ),
    }
    result["sanity"] = sanity_guards(result["splits"]["test"])
    return result


def split_metrics(split: SplitData, scores: np.ndarray, t: BandThresholds) -> dict[str, Any]:
    y = split.labels
    prevalence = m.prevalence(y)
    pr_auc = m.average_precision(y, scores)
    return {
        "rows": split.rows,
        "positives": int(y.sum()),
        "label_rate": prevalence,
        "prevalence": prevalence,  # PR-AUC of a random classifier: the reference line
        "pr_auc": pr_auc,
        "pr_auc_lift": pr_auc / prevalence if prevalence else float("nan"),
        "roc_auc": m.roc_auc(y, scores),
        "recall_at_fpr_0_1pct": m.recall_at_fpr(y, scores, 0.001),
        "recall_at_fpr_1pct": m.recall_at_fpr(y, scores, 0.01),
        "amount_weighted_recall_at_review": m.amount_weighted_recall(
            y, scores >= t.review, split.amounts
        ),
        "amount_weighted_recall_at_decline": m.amount_weighted_recall(
            y, scores >= t.decline, split.amounts
        ),
        "brier": m.brier_score(y, scores),
        "deciles": m.score_deciles(y, scores),
    }


def band_table(split: SplitData, scores: np.ndarray, t: BandThresholds) -> list[dict[str, Any]]:
    """Traffic share, precision and recall per band: the manual-review volume made visible."""
    bands = assign_bands(scores, t)
    total_positives = int(split.labels.sum())
    rows = []
    for band in BANDS:
        in_band = bands == band
        positives = int(split.labels[in_band].sum())
        rows.append(
            {
                "band": band,
                "rows": int(in_band.sum()),
                "traffic_share": float(in_band.mean()),
                "positives": positives,
                "precision": positives / in_band.sum() if in_band.any() else float("nan"),
                "recall": positives / total_positives if total_positives else float("nan"),
            }
        )
    return rows


def pattern_recall(split: SplitData, scores: np.ndarray, t: BandThresholds) -> list[dict[str, Any]]:
    """Recall per fraud pattern against the *ground truth* (analysis-only columns).

    Counts every fraudulent transaction of the pattern, including those that never got a
    chargeback label, so it shows what the model catches of the real fraud and not only of the
    labeled part.
    """
    pattern = split.frame["fraud_pattern"].to_numpy()
    rows = []
    for name in FRAUD_PATTERNS:
        mask = pattern == name
        n = int(mask.sum())
        rows.append(
            {
                "pattern": name,
                "rows": n,
                "recall_at_review": float((scores[mask] >= t.review).mean()) if n else float("nan"),
                "recall_at_decline": (
                    float((scores[mask] >= t.decline).mean()) if n else float("nan")
                ),
            }
        )
    return rows


def label_coverage(split: SplitData) -> dict[str, Any]:
    """How much real fraud has no label (declined upstream, or never reported): selection bias."""
    is_fraud = split.frame["is_fraud"].to_numpy()
    fraud = int(is_fraud.sum())
    labeled = int((is_fraud & split.labels).sum())
    return {
        "fraud_rows": fraud,
        "labeled_fraud_rows": labeled,
        "unlabeled_fraud_rows": fraud - labeled,
        "coverage": labeled / fraud if fraud else float("nan"),
        "labels_that_are_not_fraud": int((split.labels & ~is_fraud).sum()),
    }


def permutation_importance(
    data: PreparedData, model: FraudMLP, split: SplitData, repeats: int, seed: int
) -> list[dict[str, Any]]:
    """PR-AUC drop on ``split`` when one feature group is shuffled (rows permuted jointly).

    The columns of a group are permuted *together*, so correlated columns (for example the two
    velocity counts) cannot stand in for each other. A large drop means the model relies on it.
    """
    dense_index = {c: i for i, c in enumerate(data.preprocessor.dense_columns)}
    cat_index = {c: i for i, c in enumerate(CATEGORICAL_COLUMNS)}
    base = m.average_precision(split.labels, predict_scores(model, split.encoded))

    rows = []
    for group, columns in FEATURE_GROUPS.items():
        dense_cols = [dense_index[c] for c in columns if c in dense_index]
        cat_cols = [cat_index[c] for c in columns if c in cat_index]
        drops = []
        for repeat in range(repeats):
            rng = np.random.default_rng([seed, repeat])
            order = rng.permutation(len(split.encoded))
            dense = split.encoded.dense.copy()
            categorical = split.encoded.categorical.copy()
            dense[:, dense_cols] = dense[order][:, dense_cols]
            categorical[:, cat_cols] = categorical[order][:, cat_cols]
            shuffled = predict_scores(model, Encoded(dense, categorical))
            drops.append(base - m.average_precision(split.labels, shuffled))
        rows.append({"group": group, "pr_auc_drop": float(np.mean(drops)), "baseline": base})
    return sorted(rows, key=lambda r: r["pr_auc_drop"], reverse=True)


def sanity_guards(test: dict[str, Any]) -> dict[str, Any]:
    """Flags to investigate, never to tune away (see the report)."""
    lift = test["pr_auc_lift"]
    return {
        "test_pr_auc_lift": lift,
        "at_least_3x_prevalence": bool(lift >= SANITY_MIN_LIFT),
        "suspiciously_perfect": bool(test["pr_auc"] > SANITY_SUSPICIOUS_PR_AUC),
    }


def flatten_metrics(evaluation: dict[str, Any]) -> dict[str, float]:
    """The headline numbers as ``{"test.pr_auc": ...}``, as served by ``GET /v1/model``."""
    keys = (
        "prevalence",
        "pr_auc",
        "pr_auc_lift",
        "roc_auc",
        "recall_at_fpr_0_1pct",
        "recall_at_fpr_1pct",
        "amount_weighted_recall_at_review",
        "amount_weighted_recall_at_decline",
        "brier",
    )
    flat: dict[str, float] = {}
    for split in ("validation", "test"):
        for key in keys:
            value = evaluation["splits"][split][key]
            if not math.isnan(value):
                flat[f"{split}.{key}"] = float(value)
    return flat
