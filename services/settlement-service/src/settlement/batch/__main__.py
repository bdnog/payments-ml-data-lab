"""Batch execution mode CLI.

Usage::

    python -m settlement.batch run --date YYYY-MM-DD
    python -m settlement.batch run-range --from YYYY-MM-DD --to YYYY-MM-DD   # or --all
    python -m settlement.batch verify

``run`` and ``run-range`` are the direct ops and backfill path: they bypass the ``runs`` table and
are idempotent because each partition is overwritten atomically. ``verify`` compares the breaks mart
with the generator's answer key and exits non-zero unless precision and recall are 100%.
"""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path

from payments_shared import configure_logging
from settlement.batch.pipeline import RunSummary, run_dates
from settlement.config import Settings
from settlement.storage.datasets import psp_date_range

logger = logging.getLogger("settlement.batch")


def _iso_date(value: str) -> date:
    """Parse a strict ``YYYY-MM-DD`` date for argparse."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}, expected YYYY-MM-DD") from None


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-dir", type=Path, help="dataset root (default SETTLEMENT_INPUT_DIR)"
    )
    parser.add_argument("--marts-dir", type=Path, help="marts root (default SETTLEMENT_MARTS_DIR)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m settlement.batch")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="settle one business date (ops path, no runs table)")
    run.add_argument("--date", type=_iso_date, required=True, help="business date, YYYY-MM-DD")
    _add_paths(run)
    run.add_argument("--skip-input-validation", action="store_true")

    rng = commands.add_parser("run-range", help="settle many dates in a single pass")
    group = rng.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="every date the PSP file covers")
    group.add_argument("--from", dest="start", type=_iso_date, help="first date, YYYY-MM-DD")
    rng.add_argument("--to", dest="end", type=_iso_date, help="last date (with --from)")
    _add_paths(rng)
    rng.add_argument("--skip-input-validation", action="store_true")

    verify = commands.add_parser(
        "verify", help="compare the breaks mart with the generator's answer key"
    )
    _add_paths(verify)

    worker = commands.add_parser("worker", help="execute queued runs from the runs table")
    worker.add_argument("--once", action="store_true", help="drain the queue, then exit")
    return parser


def _print_summary(summary: RunSummary) -> None:
    print(json.dumps(summary.to_dict(), indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    configure_logging(settings.log_level)

    if args.command == "worker":
        from settlement.batch.worker import run_worker

        return run_worker(settings, once=args.once)

    input_dir = args.input_dir or settings.resolved_input_dir()
    marts_dir = args.marts_dir or settings.resolved_marts_dir()

    if args.command == "verify":
        from settlement.batch.verification import verify

        outcome = verify(input_dir, marts_dir)
        print(outcome.render())
        return 0 if outcome.ok else 1

    if args.command == "run":
        dates = [args.date]
    elif args.all:
        first, last = psp_date_range(input_dir)
        dates = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    else:
        if args.start is None or args.end is None:
            build_parser().error("run-range needs --all, or both --from and --to")
        if args.end < args.start:
            build_parser().error("--to must not be before --from")
        dates = [args.start + timedelta(days=i) for i in range((args.end - args.start).days + 1)]

    summary = run_dates(dates, input_dir, marts_dir, validate=not args.skip_input_validation)
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
