"""In-process latency benchmark of the scorer: N single-request scorings, p50 / p95 / p99.

Measures what ``latency_ms`` in the API response measures: feature computation plus inference for
one request, using the promoted model (``models/current``) and rows of the committed sample. It
does *not* include HTTP, JSON parsing, validation or the database, so the end-to-end latency of the
API is higher; use it to see the cost of the model itself.

Usage: ``uv run python scripts/bench_risk.py [--requests 1000] [--threads 1]``.
"""

import argparse
import statistics
import time

import polars as pl

from payments_shared.paths import repo_root
from risk_scoring.config import Settings
from risk_scoring.dataset_mapping import row_to_request
from risk_scoring.serving.scorer import Scorer

WARMUP = 50


def percentile(sorted_values: list[float], q: float) -> float:
    index = min(len(sorted_values) - 1, round(q * (len(sorted_values) - 1)))
    return sorted_values[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--requests", type=int, default=1_000)
    parser.add_argument("--threads", type=int, default=1, help="torch threads (RISK_TORCH_THREADS)")
    args = parser.parse_args()

    settings = Settings(torch_threads=args.threads)
    scorer = Scorer.load(settings.resolved_model_dir(), settings)
    rows = pl.read_parquet(repo_root() / "data" / "samples" / "risk" / "transactions.parquet")
    requests = [row_to_request(r) for r in rows.head(args.requests + WARMUP).iter_rows(named=True)]

    for request in requests[:WARMUP]:  # the first calls pay one-off costs (allocations, caches)
        scorer.score_one(request)

    timings_ms: list[float] = []
    for request in requests[WARMUP:]:
        started = time.perf_counter()
        scorer.score_one(request)
        timings_ms.append((time.perf_counter() - started) * 1000)

    timings_ms.sort()
    p95 = percentile(timings_ms, 0.95)
    print(
        f"model {scorer.model_version}, {len(timings_ms)} single requests, {args.threads} thread(s)"
    )
    print(
        f"p50 {percentile(timings_ms, 0.50):.2f} ms   p95 {p95:.2f} ms   "
        f"p99 {percentile(timings_ms, 0.99):.2f} ms   mean {statistics.fmean(timings_ms):.2f} ms"
    )
    print(f"target p95 < 20 ms: {'met' if p95 < 20 else 'NOT MET'}")


if __name__ == "__main__":
    main()
