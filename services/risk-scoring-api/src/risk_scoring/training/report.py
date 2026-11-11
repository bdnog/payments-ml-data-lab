"""Render ``report.md`` from an evaluation. Deterministic: no wall-clock time in the output."""

import json
from typing import Any

from risk_scoring.training.dataset import PreparedData
from risk_scoring.training.preprocessing import Preprocessor
from risk_scoring.training.thresholds import BandThresholds
from risk_scoring.training.trainer import TrainConfig, TrainResult

DISCLAIMER = (
    "**The data is synthetic.** These numbers describe how well a model separates the fraud "
    "patterns that *our own generator* produced. They say nothing about real-world fraud "
    "performance: the generator probably makes the problem easier (clean, stationary patterns) "
    "and harder (hand-made overlap) than reality in ways nobody can quantify."
)


def pct(value: float, digits: int = 2) -> str:
    return "n/a" if value != value else f"{100 * value:.{digits}f}%"


def num(value: float, digits: int = 4) -> str:
    return "n/a" if value != value else f"{value:.{digits}f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def render_report(
    *,
    model_version: str,
    data: PreparedData,
    evaluation: dict[str, Any],
    thresholds: BandThresholds,
    train_config: TrainConfig,
    result: TrainResult,
    preprocessor: Preprocessor,
    manifest: dict[str, Any] | None,
) -> str:
    splits, sanity = evaluation["splits"], evaluation["sanity"]
    out: list[str] = [f"# Model report: `{model_version}`", "", DISCLAIMER, ""]

    out += ["## Setup", ""]
    bounds = data.plan.to_dict()
    out += [
        f"- Split by `occurred_at` (fractions of the window): train `{bounds['window_start']}` to "
        f"`{bounds['train_end']}`, validation to `{bounds['validation_end']}`, test to "
        f"`{bounds['window_end']}`. No account-level split (production scores known accounts).",
        f"- Training labels read as of `{data.label_as_of.isoformat()}` "
        f"(train end + {data.label_maturity_days} days, the maximum chargeback delay). "
        "Validation and test use eventual labels (an offline evaluation after the fact).",
        f"- Seed {train_config.seed}; best epoch {result.best_epoch} of {len(result.history)} "
        f"(early stopping {'triggered' if result.stopped_early else 'not triggered'}); "
        f"pos_weight {result.pos_weight:.2f}.",
        "- The `score` is a **ranking score in [0, 1], not a calibrated probability** "
        "(the loss is weighted). Use the thresholds below, not the raw value as a risk percentage.",
        "",
        "Data manifest of the dataset used:",
        "",
        "```json",
        json.dumps(manifest, indent=2, sort_keys=True) if manifest else "not available",
        "```",
        "",
        "Hyperparameters:",
        "",
        "```json",
        json.dumps(train_config.to_dict(), indent=2, sort_keys=True),
        "```",
        "",
    ]

    out += ["## Splits and prevalence", ""]
    rows = [
        [
            name,
            f"{s['rows']:,}",
            f"{s['positives']:,}",
            pct(s["label_rate"], 3),
            num(s["pr_auc"]),
            f"{s['pr_auc_lift']:.1f}x",
        ]
        for name, s in splits.items()
    ]
    out += [
        table(["split", "rows", "positives", "label rate", "PR-AUC", "lift over prevalence"], rows),
        "",
        "The **prevalence** (label rate) is the PR-AUC of a random classifier: the reference line. "
        "There is no baseline model in this phase on purpose (ADR 0007).",
        "",
    ]

    out += ["## Ranking metrics", ""]
    rows = [
        [
            name,
            num(s["pr_auc"]),
            num(s["roc_auc"]),
            pct(s["recall_at_fpr_0_1pct"]),
            pct(s["recall_at_fpr_1pct"]),
            pct(s["amount_weighted_recall_at_review"]),
            pct(s["amount_weighted_recall_at_decline"]),
            num(s["brier"], 5),
        ]
        for name, s in splits.items()
    ]
    out += [
        table(
            [
                "split",
                "PR-AUC",
                "ROC-AUC",
                "recall @ FPR 0.1%",
                "recall @ FPR 1%",
                "amount recall @ review",
                "amount recall @ decline",
                "Brier",
            ],
            rows,
        ),
        "",
        "Recall is measured against the *labels* (reported chargebacks). Amount-weighted recall is "
        "the share of the labeled fraud *money* (`amount_minor`) at or above the threshold.",
        "",
    ]

    out += [
        "## Decision bands",
        "",
        f"Thresholds were chosen on the validation set: review at FPR "
        f"{pct(thresholds.review_fpr, 1)}, decline at FPR {pct(thresholds.decline_fpr, 1)} of "
        f"label-negatives. `review` = {thresholds.review:.6f}, "
        f"`decline` = {thresholds.decline:.6f}.",
        "",
    ]
    for split in ("validation", "test"):
        rows = [
            [
                b["band"],
                f"{b['rows']:,}",
                pct(b["traffic_share"], 2),
                f"{b['positives']:,}",
                pct(b["precision"]),
                pct(b["precision_ground_truth"]),
                pct(b["recall"]),
            ]
            for b in evaluation["bands"][split]
        ]
        out += [
            f"**{split}**",
            "",
            table(
                [
                    "band",
                    "rows",
                    "share of traffic",
                    "labeled fraud",
                    "precision (labels)",
                    "precision (ground truth)",
                    "recall (labels)",
                ],
                rows,
            ),
            "",
        ]

    out += ["## Recall by fraud pattern (ground truth, analysis only)", ""]
    for split in ("validation", "test"):
        rows = [
            [
                p["pattern"],
                f"{p['rows']:,}",
                pct(p["recall_at_review"]),
                pct(p["recall_at_decline"]),
            ]
            for p in evaluation["patterns"][split]
        ]
        out += [
            f"**{split}**",
            "",
            table(["pattern", "fraud rows", "recall at review", "recall at decline"], rows),
            "",
        ]

    out += [
        "## Label coverage (selection bias)",
        "",
        "Ground-truth fraud that has **no label** was declined upstream (and so never charged "
        "back) or was never reported. The model cannot learn from it, and the metrics above do "
        "not count it.",
        "",
    ]
    rows = [
        [
            name,
            f"{c['fraud_rows']:,}",
            f"{c['labeled_fraud_rows']:,}",
            f"{c['unlabeled_fraud_rows']:,}",
            pct(c["coverage"]),
        ]
        for name, c in evaluation["label_coverage"].items()
    ]
    out += [table(["split", "fraud rows", "labeled", "unlabeled", "coverage"], rows), ""]

    out += ["## Permutation importance by feature group (validation)", ""]
    rows = [[r["group"], num(r["pr_auc_drop"])] for r in evaluation["permutation_importance"]]
    out += [
        table(["feature group", "PR-AUC drop when shuffled"], rows),
        "",
        f"Baseline validation PR-AUC: {num(evaluation['permutation_importance'][0]['baseline'])}. "
        "Groups are shuffled jointly; the drop is the mean of repeated shuffles.",
        "",
    ]

    out += ["## Score deciles (test)", ""]
    rows = [
        [
            str(d["decile"]),
            f"{d['rows']:,}",
            f"{d['positives']:,}",
            pct(d["positive_rate"]),
            f"{d['min_score']:.4f}",
            f"{d['max_score']:.4f}",
        ]
        for d in splits["test"]["deciles"]
    ]
    out += [
        table(["decile (1 = highest)", "rows", "labeled fraud", "rate", "min", "max"], rows),
        "",
    ]

    out += ["## Training curve", ""]
    rows = [[str(r.epoch), f"{r.train_loss:.4f}", num(r.validation_pr_auc)] for r in result.history]
    out += [table(["epoch", "train loss", "validation PR-AUC"], rows), ""]

    out += [
        "## Sanity guards",
        "",
        f"- Test PR-AUC is {sanity['test_pr_auc_lift']:.1f}x the prevalence "
        f"(guard: at least 3x): "
        f"{'held' if sanity['at_least_3x_prevalence'] else 'NOT HELD, investigate'}.",
        f"- Test PR-AUC above 0.98 (guard: suspiciously perfect, check for leakage): "
        f"{'YES, investigate' if sanity['suspiciously_perfect'] else 'no'}.",
        "",
        "These are for investigation, not for tuning. "
        f"Vocabulary sizes: {preprocessor.vocab_sizes}.",
        "",
    ]
    return "\n".join(out)
