"""Batch execution mode CLI.

Usage::

    python -m settlement.batch run --date YYYY-MM-DD
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import date, datetime

from payments_shared import configure_logging
from settlement.config import Settings

logger = logging.getLogger("settlement.batch")


def _iso_date(value: str) -> date:
    """Parse a strict ``YYYY-MM-DD`` date for argparse."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}, expected YYYY-MM-DD") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m settlement.batch")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run the settlement batch for one business date")
    run.add_argument("--date", type=_iso_date, required=True, help="business date, YYYY-MM-DD")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(Settings().log_level)
    logger.info("settlement run requested for %s (not implemented yet)", args.date.isoformat())
    return 0


if __name__ == "__main__":
    sys.exit(main())
