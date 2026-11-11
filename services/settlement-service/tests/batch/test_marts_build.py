"""What the marts mean, and that their numbers tie out to the ledger."""

from datetime import date, timedelta

import polars as pl

from settlement.batch.pipeline import run_dates
from settlement.domain import BreakType
from settlement.storage.marts import MartStore


def test_daily_mart_is_the_ledger_expected_payout(scenario):
    scenario.payment(1, amount=10_000)  # fee 100
    scenario.payment(2, amount=5_000)  # fee 50
    scenario.refund(1, payment=1, amount=2_500)
    scenario.payment(3, amount=8_000)
    scenario.chargeback(1, payment=3, amount=8_000)
    scenario.run("2026-03-11")

    row = scenario.mart("daily_settlement_by_merchant").row(0, named=True)

    assert row == {
        "settlement_date": date(2026, 3, 11),
        "merchant_id": "mer_000001",
        "currency": "BRL",
        "payment_count": 3,
        "gross_minor": 23_000,
        "fee_minor": 230,
        "refunds_minor": 2_500,
        "chargebacks_minor": 8_000,  # the flat 1,500 PSP fee is absorbed by the platform
        "net_minor": 23_000 - 230 - 2_500 - 8_000,
    }


def test_the_payout_comes_from_the_ledger_not_from_what_the_psp_reported(scenario):
    scenario.payment(1, amount=10_000)
    scenario.psp_record(1, 1, gross=9_000, fee=90)  # the PSP disagrees

    scenario.run("2026-03-11")

    assert scenario.mart("daily_settlement_by_merchant")["gross_minor"].to_list() == [10_000]


def test_a_day_with_only_a_refund_has_a_negative_net(scenario):
    scenario.payment(1, amount=10_000, captured="2026-03-01")
    scenario.refund(1, payment=1, amount=4_000, on="2026-03-10")
    scenario.run("2026-03-11")

    row = scenario.mart("daily_settlement_by_merchant").row(0, named=True)

    assert (row["payment_count"], row["gross_minor"], row["fee_minor"]) == (0, 0, 0)
    assert (row["refunds_minor"], row["net_minor"]) == (4_000, -4_000)


def test_each_merchant_gets_its_own_row(scenario):
    scenario.merchant("mer_00000a", mdr_bps=300, settlement_days=1)
    scenario.payment(1, amount=10_000)
    scenario.payment(2, amount=10_000, merchant="mer_00000a")
    scenario.run("2026-03-11")

    daily = scenario.mart("daily_settlement_by_merchant").sort("merchant_id")
    assert daily["merchant_id"].to_list() == ["mer_000001", "mer_00000a"]
    assert daily["fee_minor"].to_list() == [100, 300]


def test_dates_with_nothing_to_report_still_get_empty_partitions(scenario):
    scenario.payment(1)
    scenario.psp_record(1, 1)
    scenario.run("2026-03-11", "2026-03-12")

    store = scenario.store
    for mart in ("daily_settlement_by_merchant", "reconciliation_breaks", "fee_summary"):
        assert store.has_partition(mart, date(2026, 3, 12)), mart
        assert store.read_partition(mart, date(2026, 3, 12)).height == 0
    assert store.read_partition("reconciliation_breaks", date(2026, 3, 11)).height == 0
    assert store.has_partition("reconciliation_breaks", date(2026, 3, 11))


def test_a_run_for_a_date_without_any_data_is_an_empty_run(scenario):
    scenario.payment(1)
    summary = scenario.run("2030-01-01")

    assert summary.rows == {
        "daily_settlement_by_merchant": 0,
        "fee_summary": 0,
        "reconciliation_breaks": 0,
    }
    assert summary.expected_payout_minor == 0


def test_break_ids_are_deterministic_unique_and_sorted(scenario):
    scenario.payment(1)
    scenario.payment(2)
    scenario.psp_record(1, None)
    scenario.psp_record(2, None, on="2026-03-12")
    scenario.run("2026-03-11", "2026-03-12")
    first = scenario.breaks()
    scenario.run("2026-03-11", "2026-03-12")
    second = scenario.breaks()

    assert first["break_id"].to_list() == second["break_id"].to_list()
    assert first["break_id"].n_unique() == first.height
    assert first["break_id"].to_list() == sorted(first["break_id"].to_list())
    assert all(i.startswith("brk_") and len(i) == 12 for i in first["break_id"])


# --- totals tie out on a real dataset ---------------------------------------------------------


def run_whole_sample(sample_dir, tmp_path) -> MartStore:
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(30)]
    run_dates(days, sample_dir, tmp_path / "marts")
    return MartStore(tmp_path / "marts")


def test_gross_ties_out_to_the_captured_payments_due_in_the_range(sample_dir, tmp_path):
    store = run_whole_sample(sample_dir, tmp_path)
    merchants = pl.read_parquet(sample_dir / "reference" / "merchants.parquet")
    payments = pl.read_parquet(sample_dir / "settlement" / "payments.parquet")

    due = (
        payments.filter(pl.col("status") == "CAPTURED")
        .join(merchants.select("merchant_id", "settlement_days"), on="merchant_id")
        .with_columns(
            due=pl.col("captured_at").dt.date() + pl.duration(days=pl.col("settlement_days"))
        )
        .filter(pl.col("due") <= date(2026, 1, 30))
    )
    daily = store.scan("daily_settlement_by_merchant").collect()

    assert daily["gross_minor"].sum() == due["amount_minor"].sum()
    assert daily["payment_count"].sum() == due.height


def test_net_is_gross_minus_fee_refunds_and_chargebacks_on_every_row(sample_dir, tmp_path):
    daily = run_whole_sample(sample_dir, tmp_path).scan("daily_settlement_by_merchant").collect()
    check = daily.select(
        (
            pl.col("net_minor")
            == pl.col("gross_minor")
            - pl.col("fee_minor")
            - pl.col("refunds_minor")
            - pl.col("chargebacks_minor")
        ).all()
    ).item()
    assert check and (daily["net_minor"] < 0).any(), "the sample has days with a negative net"


def test_the_fee_difference_per_day_equals_the_fee_mismatch_diffs(sample_dir, tmp_path):
    store = run_whole_sample(sample_dir, tmp_path)
    fees = store.scan("fee_summary").collect()
    breaks = store.scan("reconciliation_breaks").collect()

    by_day_fees = (
        fees.group_by("settlement_date").agg(pl.col("fee_diff_minor").sum()).sort("settlement_date")
    )
    by_day_breaks = (
        breaks.filter(pl.col("break_type") == BreakType.FEE_MISMATCH)
        .group_by("settlement_date")
        .agg(pl.col("diff_minor").sum().alias("fee_diff_minor"))
        .sort("settlement_date")
    )
    nonzero = by_day_fees.filter(pl.col("fee_diff_minor") != 0)

    assert nonzero.equals(by_day_breaks)
    assert breaks.filter(pl.col("break_type") == BreakType.FEE_MISMATCH).height > 0


def test_every_date_of_the_range_has_all_three_partitions(sample_dir, tmp_path):
    store = run_whole_sample(sample_dir, tmp_path)
    expected = [date(2026, 1, 1) + timedelta(days=i) for i in range(30)]
    for mart in ("daily_settlement_by_merchant", "reconciliation_breaks", "fee_summary"):
        assert store.partition_dates(mart) == expected


def test_a_run_for_one_date_equals_the_same_date_inside_a_range(sample_dir, tmp_path):
    """The pipeline is date-agnostic: the answer for a date does not depend on its neighbours."""
    store_all = run_whole_sample(sample_dir, tmp_path / "all")
    run_dates([date(2026, 1, 20)], sample_dir, tmp_path / "one" / "marts")
    store_one = MartStore(tmp_path / "one" / "marts")
    for mart in ("daily_settlement_by_merchant", "reconciliation_breaks", "fee_summary"):
        assert store_one.read_partition(mart, date(2026, 1, 20)).equals(
            store_all.read_partition(mart, date(2026, 1, 20))
        )


def test_rerunning_is_idempotent(sample_dir, tmp_path):
    store = run_whole_sample(sample_dir, tmp_path)
    before = {
        mart: store.scan(mart).collect().sort(store.scan(mart).collect_schema().names())
        for mart in ("daily_settlement_by_merchant", "reconciliation_breaks", "fee_summary")
    }
    run_whole_sample(sample_dir, tmp_path)
    for mart, frame in before.items():
        again = store.scan(mart).collect()
        assert again.sort(again.columns).equals(frame), mart
