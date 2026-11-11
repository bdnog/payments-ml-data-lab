"""The end-to-end training run: parquet in, artifact bundle out. Used by the CLI and by tests."""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from risk_scoring.serving.reasons import ReasonRules
from risk_scoring.training.artifacts import (
    METRICS_FILE,
    REPORT_FILE,
    Bundle,
    load_bundle,
    save_bundle,
)
from risk_scoring.training.dataset import PreparedData, prepare_data
from risk_scoring.training.evaluate import evaluate, flatten_metrics
from risk_scoring.training.labels import DEFAULT_LABEL_MATURITY_DAYS
from risk_scoring.training.report import render_report
from risk_scoring.training.splits import DEFAULT_FRACTIONS
from risk_scoring.training.thresholds import (
    DEFAULT_DECLINE_FPR,
    DEFAULT_REVIEW_FPR,
    choose_thresholds,
)
from risk_scoring.training.trainer import (
    EpochRecord,
    TrainConfig,
    predict_scores,
    train_model,
)

TRANSACTIONS_PATH = "risk/transactions.parquet"
MANIFEST_PATH = "manifest.json"


@dataclass(frozen=True)
class RunSummary:
    out_dir: Path
    model_version: str
    seconds: float
    evaluation: dict[str, Any]
    data: PreparedData


def read_transactions(data_dir: Path) -> pl.DataFrame:
    path = data_dir / TRANSACTIONS_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Generate the data first: "
            "`uv run python -m payments_datagen generate --profile full --out data/out/full`"
        )
    return pl.read_parquet(path)


def read_manifest(data_dir: Path) -> dict[str, Any] | None:
    path = data_dir / MANIFEST_PATH
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {k: manifest[k] for k in ("profile", "parameters", "realized") if k in manifest}


def run_training(
    data_dir: Path,
    out_dir: Path,
    model_version: str,
    *,
    train_config: TrainConfig | None = None,
    trained_at: datetime | None = None,
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
    label_maturity_days: int = DEFAULT_LABEL_MATURITY_DAYS,
    review_fpr: float = DEFAULT_REVIEW_FPR,
    decline_fpr: float = DEFAULT_DECLINE_FPR,
    reason_rules: ReasonRules | None = None,
    min_category_count: int = 20,
    on_epoch: Callable[[EpochRecord], None] | None = None,
) -> RunSummary:
    started = time.perf_counter()
    train_config = train_config or TrainConfig()
    reason_rules = reason_rules or ReasonRules()
    manifest = read_manifest(data_dir)

    data = prepare_data(
        read_transactions(data_dir),
        fractions=fractions,
        label_maturity_days=label_maturity_days,
        min_category_count=min_category_count,
    )
    train, validation = data.splits["train"], data.splits["validation"]
    result = train_model(
        train.encoded,
        train.labels,
        validation.encoded,
        validation.labels,
        data.preprocessor.vocab_sizes,
        train_config,
        on_epoch,
    )
    thresholds = choose_thresholds(
        validation.labels, predict_scores(result.model, validation.encoded), review_fpr, decline_fpr
    )
    evaluation = evaluate(data, result.model, thresholds, seed=train_config.seed)

    report = render_report(
        model_version=model_version,
        data=data,
        evaluation=evaluation,
        thresholds=thresholds,
        train_config=train_config,
        result=result,
        preprocessor=data.preprocessor,
        manifest=manifest,
    )
    save_bundle(
        out_dir,
        model=result.model,
        preprocessor=data.preprocessor,
        thresholds=thresholds,
        reason_rules=reason_rules,
        train_config=train_config,
        model_version=model_version,
        trained_at=trained_at or datetime.now(UTC),
        flat_metrics=flatten_metrics(evaluation),
        extra={
            "split": data.plan.to_dict(),
            "split_fractions": list(fractions),
            "label_as_of": data.label_as_of.isoformat(),
            "label_maturity_days": label_maturity_days,
            "best_epoch": result.best_epoch,
            "pos_weight": result.pos_weight,
            "data_manifest": manifest,
            "synthetic_data_notice": "Synthetic data: metrics say nothing about real-world fraud.",
        },
    )
    (out_dir / REPORT_FILE).write_text(report, encoding="utf-8")
    (out_dir / METRICS_FILE).write_text(
        json.dumps(evaluation, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return RunSummary(out_dir, model_version, time.perf_counter() - started, evaluation, data)


def run_evaluation(bundle_dir: Path, data_dir: Path) -> tuple[Bundle, dict[str, Any]]:
    """Evaluate a saved artifact on a dataset, with the artifact's own preprocessor and bands."""
    bundle = load_bundle(bundle_dir)
    meta = bundle.metadata
    data = prepare_data(
        read_transactions(data_dir),
        fractions=tuple(meta["split_fractions"]),  # type: ignore[arg-type]
        label_maturity_days=meta["label_maturity_days"],
        preprocessor=bundle.preprocessor,
    )
    return bundle, evaluate(data, bundle.model, bundle.thresholds, seed=bundle.train_config.seed)


def _json_default(value: object) -> object:
    if isinstance(value, float) and value != value:  # NaN is not valid JSON
        return None
    raise TypeError(f"not JSON serializable: {type(value)}")
