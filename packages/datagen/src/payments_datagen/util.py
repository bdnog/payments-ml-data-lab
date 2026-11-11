"""Small helpers shared by the generation stages: UTC timestamps and hexadecimal ids."""

from datetime import datetime

import numpy as np
import polars as pl

US_PER_SECOND = 1_000_000
US_PER_MINUTE = 60 * US_PER_SECOND
US_PER_HOUR = 60 * US_PER_MINUTE
US_PER_DAY = 24 * US_PER_HOUR

TS_DTYPE = pl.Datetime("us", "UTC")


def epoch_us(moment: datetime) -> int:
    """Microseconds since the epoch for a timezone-aware datetime on a whole second."""
    return int(moment.timestamp()) * US_PER_SECOND


def ts_series(name: str, microseconds: np.ndarray) -> pl.Series:
    """A ``timestamp_utc`` Series from epoch microseconds."""
    values = pl.Series(name, np.asarray(microseconds, dtype=np.int64), dtype=pl.Int64)
    return values.cast(pl.Datetime("us")).dt.replace_time_zone("UTC")


def epoch_us_of(series: pl.Series) -> np.ndarray:
    """Epoch microseconds (int64) of a ``timestamp_utc`` Series."""
    return series.dt.epoch("us").to_numpy()


def hex_ids(
    rng: np.random.Generator, prefix: str, hex_len: int, n: int, name: str = "id"
) -> pl.Series:
    """``n`` unique ids ``<prefix><hex_len lowercase hex digits>``, sorted ascending.

    Rows must already be in creation/time order: handing the ids out in sorted order makes them grow
    with time, like real sequence-ish identifiers, while still looking like opaque hex.
    """
    if n == 0:
        return pl.Series(name, [], dtype=pl.String)
    space = 16**hex_len
    if n > space // 4:
        raise ValueError(f"cannot draw {n} unique ids from {hex_len} hex digits")
    unique = np.unique(rng.integers(0, space, size=n + n // 8 + 16, dtype=np.int64))
    while len(unique) < n:
        extra = rng.integers(0, space, size=n, dtype=np.int64)
        unique = np.unique(np.concatenate([unique, extra]))
    chosen = (
        unique if len(unique) == n else unique[np.sort(rng.choice(len(unique), n, replace=False))]
    )
    return pl.Series(
        name, np.char.add(prefix, np.char.mod(f"%0{hex_len}x", chosen)), dtype=pl.String
    )
