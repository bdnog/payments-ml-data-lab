"""Command line: ``python -m risk_scoring.training train|evaluate``."""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from risk_scoring.training.console import print_summary
from risk_scoring.training.pipeline import run_evaluation, run_training
from risk_scoring.training.trainer import EpochRecord, TrainConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m risk_scoring.training")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train a model and write an artifact bundle")
    train.add_argument("--data", type=Path, required=True, help="datagen output directory")
    train.add_argument("--out", type=Path, required=True, help="directory of the run to write")
    train.add_argument("--model-version", required=True, help="explicit, e.g. mlp-2026.11.1")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--max-epochs", type=int, default=None)
    train.add_argument(
        "--trained-at",
        default=None,
        help="ISO timestamp with timezone to record (default: now, UTC)",
    )

    evaluate = commands.add_parser("evaluate", help="evaluate a saved artifact on a dataset")
    evaluate.add_argument("--model-dir", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "train":
        overrides = {"seed": args.seed}
        if args.max_epochs is not None:
            overrides["max_epochs"] = args.max_epochs
        config = TrainConfig(**overrides)
        trained_at = datetime.fromisoformat(args.trained_at) if args.trained_at else None
        if trained_at is not None and trained_at.tzinfo is None:
            trained_at = trained_at.replace(tzinfo=UTC)

        def log_epoch(record: EpochRecord) -> None:
            print(
                f"epoch {record.epoch:2d}  loss {record.train_loss:.4f}  "
                f"val PR-AUC {record.validation_pr_auc:.4f}",
                flush=True,
            )

        summary = run_training(
            args.data,
            args.out,
            args.model_version,
            train_config=config,
            trained_at=trained_at,
            on_epoch=log_epoch,
        )
        print_summary(summary.evaluation)
        print(f"\nwrote {summary.out_dir} in {summary.seconds:.1f}s")
        return 0

    _, evaluation = run_evaluation(args.model_dir, args.data)
    print_summary(evaluation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
