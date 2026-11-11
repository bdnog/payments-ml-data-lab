"""A compact console summary of an evaluation."""

from typing import Any


def print_summary(evaluation: dict[str, Any]) -> None:
    print("\nsplit         rows  positives  prevalence  PR-AUC  lift  ROC-AUC  R@0.1%  R@1%")
    for name, s in evaluation["splits"].items():
        print(
            f"{name:<11}{s['rows']:>8,}{s['positives']:>11,}{s['prevalence']:>12.4%}"
            f"{s['pr_auc']:>8.4f}{s['pr_auc_lift']:>6.1f}x{s['roc_auc']:>8.4f}"
            f"{s['recall_at_fpr_0_1pct']:>8.1%}{s['recall_at_fpr_1pct']:>7.1%}"
        )
    print(
        "\nbands (test): "
        + ", ".join(
            f"{b['band']} {b['traffic_share']:.2%} "
            f"(precision {b['precision']:.1%}, recall {b['recall']:.1%})"
            for b in evaluation["bands"]["test"]
        )
    )
    print(
        "permutation importance (validation PR-AUC drop): "
        + ", ".join(
            f"{r['group']} {r['pr_auc_drop']:.3f}" for r in evaluation["permutation_importance"]
        )
    )
    sanity = evaluation["sanity"]
    print(
        f"sanity: lift {sanity['test_pr_auc_lift']:.1f}x "
        f"(>=3x: {sanity['at_least_3x_prevalence']}), suspiciously perfect: "
        f"{sanity['suspiciously_perfect']}"
    )
