"""Generation profiles: the size and rates of a universe.

``sample`` is small enough to commit; ``full`` is the realistic one. ``tiny`` exists only for tests.
Every value can be overridden from the CLI.
"""

from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, timedelta
from typing import Any

DEFAULT_SEED = 42
DEFAULT_START_DATE = date(2026, 1, 1)


@dataclass(frozen=True)
class Profile:
    name: str
    n_accounts: int
    n_merchants: int
    days: int
    target_transactions: int
    fraud_rate: float
    break_rate: float
    seed: int = DEFAULT_SEED
    start_date: date = DEFAULT_START_DATE

    def __post_init__(self) -> None:
        for field in ("n_accounts", "n_merchants", "days", "target_transactions"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if not 0 <= self.fraud_rate < 0.5:
            raise ValueError("fraud_rate must be in [0, 0.5)")
        if not 0 <= self.break_rate < 0.5:
            raise ValueError("break_rate must be in [0, 0.5)")

    @property
    def window_start(self) -> datetime:
        return datetime(
            self.start_date.year, self.start_date.month, self.start_date.day, tzinfo=UTC
        )

    @property
    def window_end(self) -> datetime:
        """Exclusive end of the data window."""
        return self.window_start + timedelta(days=self.days)

    @property
    def last_day(self) -> date:
        """Last calendar day of the window; PSP records settle on or before it."""
        return self.start_date + timedelta(days=self.days - 1)

    def with_overrides(self, **overrides: Any) -> "Profile":
        """Return a copy with the given values replaced; ``None`` values are ignored."""
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["start_date"] = self.start_date.isoformat()
        return data


SAMPLE = Profile(
    name="sample",
    n_accounts=300,
    n_merchants=25,
    days=30,
    target_transactions=3_000,
    fraud_rate=0.03,
    break_rate=0.06,
)
FULL = Profile(
    name="full",
    n_accounts=5_000,
    n_merchants=200,
    days=180,
    target_transactions=500_000,
    fraud_rate=0.01,
    break_rate=0.02,
)
TINY = Profile(
    name="tiny",
    n_accounts=60,
    n_merchants=20,
    days=20,
    target_transactions=600,
    fraud_rate=0.05,
    break_rate=0.08,
)

PROFILES: dict[str, Profile] = {p.name: p for p in (SAMPLE, FULL, TINY)}
PUBLIC_PROFILES = ("sample", "full")


def get_profile(name: str, **overrides: Any) -> Profile:
    try:
        base = PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}") from None
    return base.with_overrides(**overrides)
