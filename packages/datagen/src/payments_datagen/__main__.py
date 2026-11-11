"""Command line: ``python -m payments_datagen generate|validate``."""

import argparse
import sys
import time
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

from payments_datagen.generate import generate_world
from payments_datagen.profiles import DEFAULT_SEED, DEFAULT_START_DATE, PUBLIC_PROFILES, get_profile
from payments_datagen.storage import write_world
from payments_datagen.validate import validate_directory


def _iso_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}, expected YYYY-MM-DD") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m payments_datagen")
    commands = parser.add_subparsers(dest="command", required=True)

    gen = commands.add_parser("generate", help="generate a universe of synthetic datasets")
    gen.add_argument("--profile", choices=PUBLIC_PROFILES, required=True)
    gen.add_argument("--out", type=Path, required=True, help="output directory")
    gen.add_argument("--seed", type=int, help=f"default {DEFAULT_SEED}")
    gen.add_argument("--start-date", type=_iso_date, help=f"default {DEFAULT_START_DATE}")
    gen.add_argument("--days", type=int)
    gen.add_argument("--fraud-rate", type=float)
    gen.add_argument("--break-rate", type=float)
    gen.add_argument("--n-accounts", type=int)
    gen.add_argument("--n-merchants", type=int)
    gen.add_argument("--target-transactions", type=int)

    val = commands.add_parser("validate", help="validate a generated directory")
    val.add_argument("directory", type=Path)
    val.add_argument(
        "--skip-determinism", action="store_true", help="skip the regenerate-and-compare check"
    )
    return parser


def _generate(args: argparse.Namespace) -> int:
    profile = get_profile(
        args.profile,
        seed=args.seed,
        start_date=args.start_date,
        days=args.days,
        fraud_rate=args.fraud_rate,
        break_rate=args.break_rate,
        n_accounts=args.n_accounts,
        n_merchants=args.n_merchants,
        target_transactions=args.target_transactions,
    )
    started = time.perf_counter()
    world = generate_world(profile)
    manifest = write_world(world, args.out)
    elapsed = time.perf_counter() - started

    print(f"generated profile={profile.name} seed={profile.seed} into {args.out} in {elapsed:.1f}s")
    for name, rows in manifest["row_counts"].items():
        print(f"  {name:<16} {rows:>9,} rows")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        return _generate(args)
    report = validate_directory(args.directory, determinism=not args.skip_determinism)
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
