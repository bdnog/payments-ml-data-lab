"""Polars expression equivalents of the domain rules.

``settlement.domain.rules`` defines the rules for one value at a time; these apply the very same
formulas to whole columns. ``tests/batch/test_parity.py`` checks that scalar and expression agree on
a grid of values, so the two cannot drift apart unnoticed.
"""

import polars as pl

from settlement.domain.rules import BASIS_POINTS


def expected_fee_expr(gross_minor: pl.Expr, mdr_bps: pl.Expr) -> pl.Expr:
    """``round_half_up(gross * mdr_bps / 10000)`` in integers, like the domain function."""
    return (gross_minor * mdr_bps + BASIS_POINTS // 2) // BASIS_POINTS


def settlement_date_expr(event_at: pl.Expr, days: pl.Expr | int) -> pl.Expr:
    """UTC calendar date of a timestamp column plus ``days``, like ``domain.settlement_date``."""
    utc_date = event_at.dt.convert_time_zone("UTC").dt.date()
    return (utc_date.cast(pl.Datetime("us")) + pl.duration(days=days)).cast(pl.Date)
