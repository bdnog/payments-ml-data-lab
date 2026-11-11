"""The Polars expressions must give exactly what the pure-Python domain functions give."""

from datetime import UTC, datetime, timedelta, timezone

import polars as pl

from settlement.batch.expressions import expected_fee_expr, settlement_date_expr
from settlement.domain import expected_fee_minor, settlement_date

GROSSES = [
    1,
    2,
    49,
    50,
    51,
    99,
    100,
    101,
    999,
    1_049,
    1_050,
    1_051,
    9_999,
    10_000,
    123_457,
    5_000_000,
]
MDRS = [50, 51, 75, 100, 125, 199, 250, 333, 499, 500]


def test_fee_expression_equals_the_domain_function_on_a_grid():
    grid = pl.DataFrame(
        [(g, m) for g in GROSSES for m in MDRS], schema=["gross", "mdr"], orient="row"
    )
    result = grid.with_columns(fee=expected_fee_expr(pl.col("gross"), pl.col("mdr")))

    for gross, mdr, fee in result.iter_rows():
        assert fee == expected_fee_minor(gross, mdr), (gross, mdr)


def test_fee_expression_covers_the_exact_halves():
    halves = pl.DataFrame({"gross": [1_050, 150, 2_050], "mdr": [100, 100, 100]})
    fees = halves.select(expected_fee_expr(pl.col("gross"), pl.col("mdr")))["gross"].to_list()
    assert fees == [11, 2, 21]


def test_settlement_date_expression_equals_the_domain_function_on_a_grid():
    base = datetime(2026, 2, 27, 0, 0, 0, tzinfo=UTC)
    instants = [
        base + timedelta(hours=h, minutes=m, seconds=s, microseconds=us)
        for h in (0, 1, 12, 23)
        for m, s, us in ((0, 0, 0), (59, 59, 999_999), (30, 15, 1))
    ] + [base + timedelta(days=d, seconds=-1) for d in range(0, 40, 3)]
    days = (0, 1, 2, 30)
    grid = pl.DataFrame(
        [(t, d) for t in instants for d in days],
        schema={"at": pl.Datetime("us", "UTC"), "days": pl.Int64},
        orient="row",
    )
    result = grid.with_columns(due=settlement_date_expr(pl.col("at"), pl.col("days")))

    for instant, n, due in result.iter_rows():
        assert due == settlement_date(instant, n), (instant, n)


def test_settlement_date_expression_converts_other_time_zones_to_utc_first():
    sao_paulo = timezone(timedelta(hours=-3))
    local = datetime(2026, 3, 10, 22, 30, tzinfo=sao_paulo)  # 01:30 UTC the next day
    frame = pl.DataFrame({"at": [local]}).with_columns(
        pl.col("at").dt.convert_time_zone("America/Sao_Paulo")
    )
    due = frame.select(settlement_date_expr(pl.col("at"), 1))["at"][0]
    assert due == settlement_date(local, 1)


def test_expression_accepts_a_constant_number_of_days():
    frame = pl.DataFrame({"at": [datetime(2026, 1, 1, 23, tzinfo=UTC)]})
    assert frame.select(settlement_date_expr(pl.col("at"), 1))["at"][0].isoformat() == "2026-01-02"
