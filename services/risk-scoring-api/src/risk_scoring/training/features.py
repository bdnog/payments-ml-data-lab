"""THE feature pipeline: one pure function used by training *and* serving.

``derive_features`` takes a Polars frame holding the ``role: feature`` columns of the
``transactions`` data contract (see ``required_columns``) and returns the derived frame that the
preprocessor consumes. It has no state and no dependency on the number of rows, so a 500,000-row
training frame and a 1-row frame built from one HTTP request go through exactly the same code.
That is how this service avoids training/serving skew (ADR 0007).

Derived columns, all documented here and nowhere else:

* ``log_*``: ``log1p`` of ``amount_minor``, ``account_age_days``, ``txn_count_1h``,
  ``txn_count_24h``, ``avg_amount_30d_minor`` and ``seconds_since_last_txn``. The two nullable
  behavior fields get a missing indicator (``*_missing``) and are filled with 0 *after* it.
* ``amount_to_avg_ratio``: amount divided by the 30-day average, clipped to [0, 20]; 1.0 ("as
  usual") when there is no average.
* ``hour_sin``/``hour_cos``, ``dow_sin``/``dow_cos``: hour of day and day of week of
  ``occurred_at`` in UTC, on the unit circle so that 23h is close to 0h and Sunday close to Monday.
* ``country_mismatch``: ``ip_country`` is known and differs from ``issuer_country``.
  ``ip_foreign`` / ``issuer_foreign``: the country is known and is not the home country (BR).
  ``ip_country_missing``: ``ip_country`` is null (in-person purchases).
* ``is_proxy``, ``device_is_new_for_account``, ``card_is_new_for_merchant`` as 0/1 floats, and
  ``installments`` as a number.
* ``mcc``, ``card_bin``, ``channel``, ``issuer_country``, ``ip_country`` stay strings: the
  preprocessor turns them into embedding indices. ``ip_country`` keeps its nulls.

Ids (``account_id``, ``merchant_id``, ``device_id``), the label and every ``analysis_only`` column
are not inputs and this module cannot see them: they are not in ``required_columns``.
"""

import math
from functools import cache

import polars as pl

from payments_shared.data_contracts import Role, load_contract

HOME_COUNTRY = "BR"
RATIO_CAP = 20.0

# The continuous derived columns: standardized with train-only statistics by the preprocessor.
NUMERIC_COLUMNS: tuple[str, ...] = (
    "log_amount",
    "log_account_age_days",
    "log_txn_count_1h",
    "log_txn_count_24h",
    "log_avg_amount_30d",
    "log_seconds_since_last_txn",
    "amount_to_avg_ratio",
    "installments",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
)
# 0/1 flags: used as they are (standardizing a 1% flag would just produce a z-score of 10).
BINARY_COLUMNS: tuple[str, ...] = (
    "avg_amount_30d_missing",
    "seconds_since_last_txn_missing",
    "country_mismatch",
    "ip_foreign",
    "issuer_foreign",
    "ip_country_missing",
    "is_proxy",
    "device_is_new_for_account",
    "card_is_new_for_merchant",
)
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "mcc",
    "card_bin",
    "channel",
    "issuer_country",
    "ip_country",
)

# Feature groups, for permutation importance. Every derived column belongs to exactly one group.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "amount": ("log_amount", "amount_to_avg_ratio", "installments"),
    "velocity": (
        "log_txn_count_1h",
        "log_txn_count_24h",
        "log_seconds_since_last_txn",
        "seconds_since_last_txn_missing",
    ),
    "history": ("log_avg_amount_30d", "avg_amount_30d_missing", "log_account_age_days"),
    "time_of_day": ("hour_sin", "hour_cos", "dow_sin", "dow_cos"),
    "geo": (
        "country_mismatch",
        "ip_foreign",
        "issuer_foreign",
        "ip_country_missing",
        "issuer_country",
        "ip_country",
    ),
    "device_and_card": (
        "is_proxy",
        "device_is_new_for_account",
        "card_is_new_for_merchant",
        "card_bin",
    ),
    "merchant_and_channel": ("mcc", "channel"),
}


@cache
def required_columns() -> tuple[str, ...]:
    """The raw input columns: exactly the ``role: feature`` columns of the data contract."""
    contract = load_contract("transactions")
    return tuple(c.name for c in contract.columns_with_role(Role.FEATURE))


def _log1p(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Float64).log1p()


def derive_features(raw: pl.DataFrame) -> pl.DataFrame:
    """Derive the model's input columns from the raw feature columns. Pure; see the module doc."""
    missing = [c for c in required_columns() if c not in raw.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")

    avg = pl.col("avg_amount_30d_minor")
    ip = pl.col("ip_country")
    hour = pl.col("occurred_at").dt.hour().cast(pl.Float64)
    # Polars weekday: Monday = 1 ... Sunday = 7.
    weekday = (pl.col("occurred_at").dt.weekday() - 1).cast(pl.Float64)
    hour_angle = hour * (2 * math.pi / 24)
    dow_angle = weekday * (2 * math.pi / 7)

    derived = raw.select(
        _log1p("amount_minor").alias("log_amount"),
        _log1p("account_age_days").alias("log_account_age_days"),
        _log1p("txn_count_1h").alias("log_txn_count_1h"),
        _log1p("txn_count_24h").alias("log_txn_count_24h"),
        _log1p("avg_amount_30d_minor").fill_null(0.0).alias("log_avg_amount_30d"),
        _log1p("seconds_since_last_txn").fill_null(0.0).alias("log_seconds_since_last_txn"),
        pl.when(avg.is_null() | (avg <= 0))
        .then(1.0)
        .otherwise(pl.col("amount_minor") / avg)
        .clip(0.0, RATIO_CAP)
        .alias("amount_to_avg_ratio"),
        pl.col("installments").cast(pl.Float64),
        hour_angle.sin().alias("hour_sin"),
        hour_angle.cos().alias("hour_cos"),
        dow_angle.sin().alias("dow_sin"),
        dow_angle.cos().alias("dow_cos"),
        avg.is_null().cast(pl.Float64).alias("avg_amount_30d_missing"),
        pl.col("seconds_since_last_txn")
        .is_null()
        .cast(pl.Float64)
        .alias("seconds_since_last_txn_missing"),
        (ip.is_not_null() & (ip != pl.col("issuer_country")))
        .cast(pl.Float64)
        .alias("country_mismatch"),
        (ip.is_not_null() & (ip != HOME_COUNTRY)).cast(pl.Float64).alias("ip_foreign"),
        (pl.col("issuer_country") != HOME_COUNTRY).cast(pl.Float64).alias("issuer_foreign"),
        ip.is_null().cast(pl.Float64).alias("ip_country_missing"),
        pl.col("is_proxy").cast(pl.Float64),
        pl.col("device_is_new_for_account").cast(pl.Float64),
        pl.col("card_is_new_for_merchant").cast(pl.Float64),
        *(pl.col(c) for c in CATEGORICAL_COLUMNS),
    )
    return derived
