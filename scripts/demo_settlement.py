"""Drive a running settlement-service end to end and print what happens.

1. ``GET /ready``;
2. ``POST /v1/runs`` for a date (``202`` and a ``Location`` header for a new run);
3. poll the run until a worker has executed it (``DONE``);
4. page through ``/v1/settlements`` and ``/v1/reconciliation/breaks``;
5. download an export and read it back with Polars;
6. ``POST`` the same date again: ``200`` with the same ``run_id`` (idempotent).

Usage: ``uv run python scripts/demo_settlement.py [--url URL] [--date 2026-01-15]``.
Needs the stack (``make up``): the API, the worker and Postgres. Running it twice for the same date
replays step 2 as a ``200``, because that date already has a run.
"""

import argparse
import io
import sys
import time
from collections import Counter

import httpx
import polars as pl


def section(title: str) -> None:
    print(f"\n== {title}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://localhost:8002")
    parser.add_argument("--date", default="2026-01-15", help="business date to settle")
    parser.add_argument("--timeout", type=float, default=120, help="seconds to wait for the run")
    args = parser.parse_args()

    with httpx.Client(base_url=args.url, timeout=30) as client:
        section("readiness")
        ready = client.get("/ready")
        print(f"GET /ready -> {ready.status_code} {ready.text}")
        if ready.status_code != 200:
            print("the service is not ready (is the stack up? `make up`)")
            return 1

        section(f"request a run for {args.date}")
        created = client.post("/v1/runs", json={"run_date": args.date})
        run = created.json()
        print(
            f"POST /v1/runs -> {created.status_code}, Location: {created.headers.get('location')}"
        )
        print(f"run {run['run_id']} is {run['status']}")
        if created.status_code not in (200, 202):
            return 1

        section("wait for the worker")
        deadline = time.monotonic() + args.timeout
        while run["status"] not in ("DONE", "FAILED") and time.monotonic() < deadline:
            time.sleep(0.5)
            run = client.get(f"/v1/runs/{run['run_id']}").json()
        print(f"GET /v1/runs/{run['run_id']} -> {run['status']}")
        if run["status"] != "DONE":
            print(f"the run did not finish: {run}")
            return 1
        print(f"started {run['started_at']}, finished {run['finished_at']}")

        section("settlements (ledger-expected payout), first page")
        page = client.get(
            "/v1/settlements", params={"from": args.date, "to": args.date, "limit": 3}
        )
        body = page.json()
        for item in body["items"]:
            print(
                f"{item['settlement_date']} {item['merchant_id']}: {item['payment_count']} sales, "
                f"gross {item['gross_minor']}, fee {item['fee_minor']}, refunds "
                f"{item['refunds_minor']}, chargebacks {item['chargebacks_minor']}, "
                f"net {item['net_minor']}"
            )
        print(f"next_cursor: {body['next_cursor']}")
        if not body["items"]:
            print("no settlements for that date: is the dataset mounted and does it cover it?")
            return 1

        section("reconciliation breaks of the run date, all pages")
        breaks, cursor = [], None
        while True:
            params = {"date": args.date, "limit": 20} | ({"cursor": cursor} if cursor else {})
            page = client.get("/v1/reconciliation/breaks", params=params).json()
            breaks += page["items"]
            cursor = page["next_cursor"]
            if cursor is None:
                break
        print(f"{len(breaks)} break(s): {dict(Counter(b['break_type'] for b in breaks))}")
        for item in breaks[:3]:
            print(
                f"  {item['break_id']} {item['break_type']}: ledger {item['ledger_amount_minor']}, "
                f"psp {item['psp_amount_minor']}, diff {item['diff_minor']}"
            )

        section("export a mart and read it back with Polars")
        export = client.get(f"/v1/exports/fee_summary/{args.date}")
        kind = export.headers["content-type"]
        print(f"GET /v1/exports/fee_summary/{args.date} -> {export.status_code} {kind}")
        frame = pl.read_parquet(io.BytesIO(export.content))
        print(f"{frame.height} rows, columns {frame.columns}")
        print(frame.head(3))

        section("the same date again is idempotent")
        again = client.post("/v1/runs", json={"run_date": args.date})
        same = again.json()["run_id"] == run["run_id"]
        print(f"POST /v1/runs -> {again.status_code}, same run_id: {same}")
        if again.status_code != 200 or not same:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
