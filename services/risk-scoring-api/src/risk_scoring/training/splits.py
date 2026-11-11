"""Temporal split by ``occurred_at``: train on the past, validate and test on the future.

The split is a fraction of the data window (default 55% / 15% / 30%), cut on the time axis, not on
rows. There is deliberately **no account-level split**: in production the model scores accounts it
has seen before, so a random or account-held-out split would measure something else (ADR 0008).
The price is that behavior of known accounts leaks across the boundary in the way it would in
production, which is what we want to measure.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl

DEFAULT_FRACTIONS = (0.55, 0.15, 0.30)
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitPlan:
    """The cut points. Each split is a half-open interval ``[start, end)``."""

    window_start: datetime
    train_end: datetime
    validation_end: datetime
    window_end: datetime

    def bounds(self) -> dict[str, tuple[datetime, datetime]]:
        return {
            "train": (self.window_start, self.train_end),
            "validation": (self.train_end, self.validation_end),
            "test": (self.validation_end, self.window_end),
        }

    def masks(self, transactions: pl.DataFrame) -> dict[str, pl.Series]:
        occurred = transactions["occurred_at"]
        return {
            name: (occurred >= start) & (occurred < end)
            for name, (start, end) in self.bounds().items()
        }

    def apply(self, transactions: pl.DataFrame) -> dict[str, pl.DataFrame]:
        return {name: transactions.filter(mask) for name, mask in self.masks(transactions).items()}

    def to_dict(self) -> dict[str, str]:
        return {
            "window_start": self.window_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "validation_end": self.validation_end.isoformat(),
            "window_end": self.window_end.isoformat(),
        }


def plan_split(
    transactions: pl.DataFrame, fractions: tuple[float, float, float] = DEFAULT_FRACTIONS
) -> SplitPlan:
    """Cut the data window (whole UTC days from the first to the last transaction) by fractions."""
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-9 or min(fractions) <= 0:
        raise ValueError(f"fractions must be three positive numbers that sum to 1: {fractions}")
    if transactions.height == 0:
        raise ValueError("cannot split an empty frame")

    first: datetime = transactions["occurred_at"].min()  # type: ignore[assignment]
    last: datetime = transactions["occurred_at"].max()  # type: ignore[assignment]
    start = first.replace(hour=0, minute=0, second=0, microsecond=0)
    end = last.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    span = end - start

    def cut(fraction: float) -> datetime:
        # Whole seconds keep the cut points readable in reports.
        return start + timedelta(seconds=round(span.total_seconds() * fraction))

    return SplitPlan(start, cut(fractions[0]), cut(fractions[0] + fractions[1]), end)
