"""Time the settlement pipeline on a dataset and report the size of the marts.

Reports, on the dataset given by ``--data`` (default ``data/out/full``):

* one date (the middle of the PSP range): with and without input validation, best of 3;
* the whole range in a single pass (``run-range --all``): the rows per mart and the time;
* the marts on disk: partitions and bytes per mart.

Marts are written to a temporary directory and removed afterwards.
Usage: ``uv run python scripts/bench_settlement.py [--data data/out/full]``.
"""

import argparse
import shutil
import tempfile
import time
from datetime import timedelta
from pathlib import Path

from settlement.batch.pipeline import run_dates
from settlement.storage.datasets import psp_date_range
from settlement.storage.marts import MART_NAMES, MartStore


def timed(fn, repeat: int = 1) -> float:
    best = float("inf")
    for _ in range(repeat):
        started = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - started)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/out/full"))
    args = parser.parse_args()

    first, last = psp_date_range(args.data)
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    middle = days[len(days) // 2]
    out = Path(tempfile.mkdtemp(prefix="settlement-bench-"))
    try:
        print(f"dataset {args.data}: PSP dates {first}..{last} ({len(days)} days)")
        one = timed(lambda: run_dates([middle], args.data, out / "one", validate=False), repeat=3)
        one_validated = timed(lambda: run_dates([middle], args.data, out / "one"), repeat=3)
        print(f"one date ({middle}), no input validation: {one:.2f}s (best of 3)")
        print(f"one date ({middle}), with input validation: {one_validated:.2f}s (best of 3)")

        summary = None

        def whole_range() -> None:
            nonlocal summary
            summary = run_dates(days, args.data, out / "all")

        total = timed(whole_range)
        print(f"whole range, {len(days)} dates in one pass, with validation: {total:.2f}s")
        print(f"  rows per mart: {summary.rows}")
        print(f"  breaks by type: {summary.breaks_by_type}")
        print(f"  expected payout: {summary.expected_payout_minor} minor units")

        store = MartStore(out / "all")
        total_bytes = 0
        for mart in MART_NAMES:
            files = [store.partition_file(mart, d) for d in store.partition_dates(mart)]
            size = sum(f.stat().st_size for f in files)
            total_bytes += size
            print(f"  {mart}: {len(files)} partitions, {size / 1024:.0f} KiB")
        print(f"marts on disk: {total_bytes / 1024 / 1024:.2f} MiB")
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    main()
