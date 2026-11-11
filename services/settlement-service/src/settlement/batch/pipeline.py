"""The batch pipeline: inputs in, three partitioned marts out.

``run_dates`` is date-agnostic: it reconciles every date it is given in a single pass and writes one
partition per date and mart. ``run --date D`` is a list of one date; ``run-range`` a list of many.
The pipeline never reads the generator's ``injected_breaks`` (only ``batch.verification`` does).
"""

import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from settlement.batch.expected import expected_records
from settlement.batch.marts import build_breaks_mart, build_daily_settlement, build_fee_summary
from settlement.batch.reconcile import reconcile
from settlement.domain import BreakType
from settlement.storage.datasets import scan_inputs, validate_inputs
from settlement.storage.marts import MartStore

log = logging.getLogger("settlement.batch")

DAILY, BREAKS, FEES = "daily_settlement_by_merchant", "reconciliation_breaks", "fee_summary"


@dataclass
class RunSummary:
    """What a run did, in numbers. Serialized into the ``runs.summary`` column."""

    dates: list[date]
    rows: dict[str, int] = field(default_factory=dict)  # per mart
    breaks_by_type: dict[str, int] = field(default_factory=dict)
    expected_payout_minor: int = 0  # sum of net_minor of the daily mart
    expected_gross_minor: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dates"] = [d.isoformat() for d in self.dates]
        data["date_count"] = len(self.dates)
        data["first_date"] = self.dates[0].isoformat() if self.dates else None
        data["last_date"] = self.dates[-1].isoformat() if self.dates else None
        del data["dates"]
        return data


def run_dates(
    dates: Sequence[date],
    input_dir: Path,
    marts_dir: Path,
    *,
    validate: bool = True,
) -> RunSummary:
    started = time.perf_counter()
    wanted = sorted(set(dates))
    if not wanted:
        raise ValueError("no dates to settle")
    if validate:
        validate_inputs(input_dir)

    inputs = scan_inputs(input_dir)
    in_scope = pl.col("settlement_date").is_in(pl.Series(wanted, dtype=pl.Date).implode())
    expected = expected_records(inputs).filter(in_scope)
    psp = inputs.psp.filter(in_scope)
    result = reconcile(expected, psp, inputs.payments)

    daily, fees, breaks = pl.collect_all(
        [
            build_daily_settlement(expected),
            build_fee_summary(result.matched_sales),
            result.breaks,
        ]
    )
    breaks_mart = build_breaks_mart(breaks)

    store = MartStore(marts_dir)
    summary = RunSummary(dates=wanted)
    for mart, frame in ((DAILY, daily), (FEES, fees), (BREAKS, breaks_mart)):
        parts = frame.partition_by("settlement_date", as_dict=True) if frame.height else {}
        for day in wanted:
            part = parts.get((day,))
            if part is None:
                store.write_empty_partition(mart, day)  # "ran, found nothing" stays visible
            else:
                store.write_partition(mart, day, part)
        summary.rows[mart] = frame.height

    summary.breaks_by_type = {
        kind.value: int((breaks_mart["break_type"] == kind.value).sum()) for kind in BreakType
    }
    summary.expected_payout_minor = int(daily["net_minor"].sum()) if daily.height else 0
    summary.expected_gross_minor = int(daily["gross_minor"].sum()) if daily.height else 0
    summary.seconds = time.perf_counter() - started
    log.info(
        "settled %d date(s) %s..%s: %s breaks, expected payout %d minor units, %.1fs",
        len(wanted),
        wanted[0],
        wanted[-1],
        summary.rows[BREAKS],
        summary.expected_payout_minor,
        summary.seconds,
    )
    return summary
