"""Replay committed sample transactions through a running risk-scoring-api and print what happened.

It shows, against the real service (Postgres decision log included):

1. readiness and the model in production;
2. single scoring of fraud and legitimate rows, and how well the scores separate them;
3. a batch call with one deliberately invalid item (a per-item error, the rest still scored);
4. an idempotent replay (the same body comes back);
5. a conflict (same ``transaction_id``, changed payload: ``409`` problem+json).

Usage: ``uv run python scripts/score_sample.py [--url http://localhost:8001] [--legit 400]``.
Rows keep their ids, so running it twice replays the stored decisions instead of scoring again.
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from typing import Any

import httpx
import polars as pl

from payments_shared.paths import repo_root
from risk_scoring.dataset_mapping import row_to_payload

SAMPLE = repo_root() / "data" / "samples" / "risk" / "transactions.parquet"


def as_json(payload: dict[str, Any]) -> dict[str, Any]:
    """Payload with datetimes as ISO strings, as they travel over HTTP."""
    return json.loads(json.dumps(payload, default=lambda value: value.isoformat()))


def section(title: str) -> None:
    print(f"\n== {title}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://localhost:8001")
    parser.add_argument("--fraud", type=int, default=90, help="fraud rows to score (max in sample)")
    parser.add_argument("--legit", type=int, default=400, help="legitimate rows to score")
    args = parser.parse_args()

    rows = pl.read_parquet(SAMPLE)
    fraud = rows.filter(pl.col("is_fraud")).head(args.fraud)
    legit = rows.filter(~pl.col("is_fraud")).head(args.legit)

    with httpx.Client(base_url=args.url, timeout=10) as client:
        section("readiness")
        ready = client.get("/ready")
        print(f"GET /ready -> {ready.status_code} {ready.text}")
        if ready.status_code != 200:
            print("the service is not ready (is the stack up? `make up`)")
            return 1
        info = client.get("/v1/model").json()
        print(f"model {info['model_version']}, thresholds {info['thresholds']}")
        print(f"test PR-AUC {info['metrics']['test.pr_auc']:.3f} on SYNTHETIC data")

        section(f"single scoring: {fraud.height} fraud and {legit.height} legitimate rows")
        scores: dict[str, list[float]] = {"fraud": [], "legit": []}
        decisions: dict[str, Counter] = {"fraud": Counter(), "legit": Counter()}
        latencies: list[float] = []
        first_body: dict[str, Any] | None = None
        first_payload: dict[str, Any] | None = None
        for kind, frame in (("fraud", fraud), ("legit", legit)):
            for row in frame.iter_rows(named=True):
                payload = as_json(row_to_payload(row))
                response = client.post("/v1/score", json=payload)
                response.raise_for_status()
                body = response.json()
                scores[kind].append(body["score"])
                decisions[kind][body["decision"]] += 1
                latencies.append(body["latency_ms"])
                if first_body is None:
                    first_body, first_payload = body, payload
        for kind in ("fraud", "legit"):
            print(
                f"{kind:>5}: mean score {statistics.fmean(scores[kind]):.4f}, "
                f"decisions {dict(decisions[kind])}"
            )
        ratio = statistics.fmean(scores["fraud"]) / max(statistics.fmean(scores["legit"]), 1e-9)
        print(f"fraud rows score {ratio:.0f}x higher on average than legitimate rows")
        print(f"mean server-side latency_ms {statistics.fmean(latencies):.2f}")

        section("batch of 100 with one invalid item")
        batch_rows = rows.filter(~pl.col("is_fraud")).slice(args.legit, 100)
        items = [as_json(row_to_payload(r)) for r in batch_rows.iter_rows(named=True)]
        items[7]["amount_minor"] = 0  # invalid: must be positive
        response = client.post("/v1/score:batch", json={"items": items})
        results = response.json()["results"]
        ok = sum("result" in r for r in results)
        errors = [r for r in results if "error" in r]
        print(
            f"POST /v1/score:batch -> {response.status_code}: {ok} scored, {len(errors)} error(s)"
        )
        for item in errors:
            print(f"  {item['transaction_id']}: {item['error']['title']} {item['error']['errors']}")

        assert first_body is not None and first_payload is not None
        section("idempotent replay")
        replay = client.post("/v1/score", json=first_payload)
        same = replay.status_code == 200 and replay.json() == first_body
        print(f"same request again -> {replay.status_code}, identical body: {same}")
        if not same:
            return 1

        section("same transaction_id, changed payload")
        changed = {**first_payload, "amount_minor": first_payload["amount_minor"] + 1}
        conflict = client.post("/v1/score", json=changed)
        print(f"-> {conflict.status_code} {conflict.headers['content-type']}")
        print(json.dumps(conflict.json(), indent=2))
        if conflict.status_code != 409:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
