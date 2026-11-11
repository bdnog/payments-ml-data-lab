"""Verification: compare the ``reconciliation_breaks`` mart with the generator's answer key.

This is the only production module that reads ``settlement/injected_breaks.parquet``, and the
pipeline never imports it: a pipeline that peeked at the answers would prove nothing. A break is a
*true positive* when the mart has a row with the same ``(break_type, payment_id, psp_record_id)``
(nulls equal) **and** the same difference (``diff_minor`` against ``expected_diff_minor``).

* precision of a type = true positives / breaks found of that type (did we invent any?),
* recall of a type    = true positives / breaks injected of that type (did we miss any?).
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import polars as pl

from settlement.batch.expressions import settlement_date_expr
from settlement.domain import BreakType
from settlement.storage.datasets import input_file
from settlement.storage.marts import MartStore

INJECTED_PATH = "settlement/injected_breaks.parquet"
KEY = ["break_type", "payment_id", "psp_record_id"]
EXAMPLES = 5


@dataclass(frozen=True)
class TypeResult:
    break_type: str
    injected: int
    found: int
    true_positives: int

    @property
    def precision(self) -> float | None:
        return self.true_positives / self.found if self.found else None

    @property
    def recall(self) -> float | None:
        return self.true_positives / self.injected if self.injected else None

    @property
    def perfect(self) -> bool:
        return self.true_positives == self.injected == self.found


@dataclass
class Verification:
    dates: list[date]
    per_type: list[TypeResult] = field(default_factory=list)
    unexpected: list[dict] = field(default_factory=list)  # in the mart, not in the answer key
    missed: list[dict] = field(default_factory=list)  # in the answer key, not in the mart

    @property
    def ok(self) -> bool:
        return bool(self.dates) and all(r.perfect for r in self.per_type)

    def render(self) -> str:
        lines = [
            f"verified {len(self.dates)} date(s): {self.dates[0]}..{self.dates[-1]}"
            if self.dates
            else "no partitions of reconciliation_breaks found: nothing to verify",
            f"{'break type':<26}{'injected':>9}{'found':>7}{'correct':>9}"
            f"{'precision':>11}{'recall':>8}",
        ]
        for r in self.per_type:
            lines.append(
                f"{r.break_type:<26}{r.injected:>9}{r.found:>7}{r.true_positives:>9}"
                f"{_pct(r.precision):>11}{_pct(r.recall):>8}"
            )
        for title, rows in (
            ("found but not injected", self.unexpected),
            ("injected but missed", self.missed),
        ):
            if rows:
                lines.append(f"{title} (first {min(len(rows), EXAMPLES)} of {len(rows)}):")
                lines += [f"  {row}" for row in rows[:EXAMPLES]]
        lines.append(
            "RESULT: " + ("100% precision and recall for every break type" if self.ok else "FAILED")
        )
        return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def injected_in_dates(input_dir: Path, dates: list[date]) -> pl.DataFrame:
    """The answer key restricted to the dates the marts cover.

    ``injected_breaks`` has no date column, so each break's date is derived from the data it
    points at: its PSP record's settlement date, or for ``MISSING_IN_PSP`` (no PSP record) the
    due date of the payment.
    """
    injected = pl.read_parquet(Path(input_dir) / INJECTED_PATH)
    psp = pl.read_parquet(input_file(input_dir, "psp_settlement")).select(
        "psp_record_id", pl.col("settlement_date").alias("psp_date")
    )
    payments = pl.read_parquet(input_file(input_dir, "payments")).select(
        "payment_id", "captured_at", "merchant_id"
    )
    merchants = pl.read_parquet(input_file(input_dir, "merchants")).select(
        "merchant_id", "settlement_days"
    )
    due = payments.join(merchants, on="merchant_id").select(
        "payment_id",
        settlement_date_expr(pl.col("captured_at"), pl.col("settlement_days")).alias("due_date"),
    )
    dated = (
        injected.join(psp, on="psp_record_id", how="left")
        .join(due, on="payment_id", how="left")
        .with_columns(pl.coalesce("psp_date", "due_date").alias("settlement_date"))
    )
    return dated.filter(pl.col("settlement_date").is_in(pl.Series(dates, dtype=pl.Date).implode()))


def verify(input_dir: Path, marts_dir: Path) -> Verification:
    store = MartStore(marts_dir)
    dates = store.partition_dates("reconciliation_breaks")
    result = Verification(dates=dates)
    if not dates:
        return result

    found = store.scan("reconciliation_breaks").collect()
    injected = injected_in_dates(input_dir, dates)

    matched = found.join(
        injected,
        on=KEY,
        how="inner",
        nulls_equal=True,
    ).filter(pl.col("diff_minor") == pl.col("expected_diff_minor"))
    true_positive = matched.group_by("break_type").len().rename({"len": "tp"})

    counts = lambda df, name: df.group_by("break_type").len().rename({"len": name})  # noqa: E731
    table = (
        pl.DataFrame({"break_type": [b.value for b in BreakType]})
        .join(counts(injected, "injected"), on="break_type", how="left")
        .join(counts(found, "found"), on="break_type", how="left")
        .join(true_positive, on="break_type", how="left")
        .fill_null(0)
    )
    result.per_type = [
        TypeResult(r["break_type"], r["injected"], r["found"], r["tp"])
        for r in table.iter_rows(named=True)
    ]

    keyed = [*KEY]
    result.unexpected = (
        found.join(matched.select(keyed), on=keyed, how="anti", nulls_equal=True)
        .select(*keyed, "diff_minor")
        .to_dicts()
    )
    result.missed = (
        injected.join(matched.select(keyed), on=keyed, how="anti", nulls_equal=True)
        .select(*keyed, "expected_diff_minor")
        .to_dicts()
    )
    return result
