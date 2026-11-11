"""One hand-built situation per break type, plus the cases that must NOT be breaks."""

from datetime import date

import polars as pl
import pytest

from settlement.domain import BreakType


def only(breaks: pl.DataFrame) -> dict:
    assert breaks.height == 1, breaks
    return breaks.row(0, named=True)


def test_a_clean_match_has_no_break(scenario):
    scenario.payment(1)
    scenario.psp_record(1, 1)

    scenario.run("2026-03-11")

    assert scenario.breaks().height == 0
    fees = scenario.mart("fee_summary")
    assert fees.row(0, named=True) == {
        "settlement_date": date(2026, 3, 11),
        "merchant_id": "mer_000001",
        "currency": "BRL",
        "gross_minor": 10_000,
        "expected_fee_minor": 100,
        "actual_fee_minor": 100,
        "fee_diff_minor": 0,
    }


def test_amount_mismatch(scenario):
    scenario.payment(1, amount=10_000)
    scenario.psp_record(1, 1, gross=9_000, fee=90)

    scenario.run("2026-03-11")

    row = only(scenario.breaks())
    assert row["break_type"] == BreakType.AMOUNT_MISMATCH
    assert (row["ledger_amount_minor"], row["psp_amount_minor"], row["diff_minor"]) == (
        10_000,
        9_000,
        -1_000,
    )
    assert row["payment_id"] == "pay_000000000001" and row["psp_record_id"] == "psp_000000000001"
    assert row["settlement_date"] == row["run_date"] == date(2026, 3, 11)


def test_amount_mismatch_takes_precedence_over_fee_mismatch(scenario):
    scenario.payment(1, amount=10_000)
    scenario.psp_record(1, 1, gross=9_000, fee=777)  # the amount AND the fee are wrong

    scenario.run("2026-03-11")

    assert only(scenario.breaks())["break_type"] == BreakType.AMOUNT_MISMATCH
    assert scenario.mart("fee_summary").height == 0, "an amount break is not a fee break"


def test_fee_mismatch_compares_fees_and_feeds_the_fee_summary(scenario):
    scenario.payment(1, amount=10_000)  # contracted fee: 100
    scenario.psp_record(1, 1, gross=10_000, fee=130)

    scenario.run("2026-03-11")

    row = only(scenario.breaks())
    assert row["break_type"] == BreakType.FEE_MISMATCH
    assert (row["ledger_amount_minor"], row["psp_amount_minor"], row["diff_minor"]) == (
        100,
        130,
        30,
    )
    fees = scenario.mart("fee_summary").row(0, named=True)
    assert (fees["expected_fee_minor"], fees["actual_fee_minor"], fees["fee_diff_minor"]) == (
        100,
        130,
        30,
    )


def test_fee_uses_the_merchants_own_rate_and_rounds_half_up(scenario):
    scenario.merchant("mer_00000a", mdr_bps=250, settlement_days=1)
    scenario.payment(1, amount=1_010, merchant="mer_00000a")  # 25.25 -> 25
    scenario.payment(2, amount=1_020, merchant="mer_00000a")  # 25.5 -> 26
    scenario.psp_record(1, 1, gross=1_010, fee=25, merchant="mer_00000a")
    scenario.psp_record(2, 2, gross=1_020, fee=25, merchant="mer_00000a")  # rounded down: wrong

    scenario.run("2026-03-11")

    row = only(scenario.breaks())
    assert row["break_type"] == BreakType.FEE_MISMATCH
    assert row["payment_id"] == "pay_000000000002"
    assert (row["ledger_amount_minor"], row["psp_amount_minor"], row["diff_minor"]) == (26, 25, -1)


def test_missing_in_psp(scenario):
    scenario.payment(1, amount=10_000)

    scenario.run("2026-03-11")

    row = only(scenario.breaks())
    assert row["break_type"] == BreakType.MISSING_IN_PSP
    assert row["payment_id"] == "pay_000000000001" and row["psp_record_id"] is None
    assert (row["ledger_amount_minor"], row["psp_amount_minor"], row["diff_minor"]) == (
        10_000,
        None,
        -10_000,
    )


@pytest.mark.parametrize("reference", [None, "pay_00000000ffff"], ids=["null-ref", "unknown-ref"])
def test_missing_in_ledger_has_no_payment_when_the_reference_is_not_a_payment(scenario, reference):
    scenario.psp_record(7, reference, gross=4_200)

    scenario.run("2026-03-11")

    row = only(scenario.breaks())
    assert row["break_type"] == BreakType.MISSING_IN_LEDGER
    assert row["payment_id"] is None and row["psp_record_id"] == "psp_000000000007"
    assert (row["ledger_amount_minor"], row["psp_amount_minor"], row["diff_minor"]) == (
        None,
        4_200,
        4_200,
    )


def test_several_orphans_with_a_null_reference_are_not_duplicates_of_each_other(scenario):
    for n in (1, 2, 3):
        scenario.psp_record(n, None, gross=1_000)

    scenario.run("2026-03-11")

    breaks = scenario.breaks()
    assert breaks["break_type"].to_list() == [BreakType.MISSING_IN_LEDGER] * 3


def test_chargeback_not_in_ledger(scenario):
    scenario.payment(1, amount=10_000)
    scenario.psp_record(1, 1)  # the sale itself is fine
    scenario.psp_record(2, 1, record_type="CHARGEBACK", gross=10_000, on="2026-03-20")

    scenario.run("2026-03-11", "2026-03-20")

    row = only(scenario.breaks())
    assert row["break_type"] == BreakType.CHARGEBACK_NOT_IN_LEDGER
    assert row["payment_id"] == "pay_000000000001" and row["psp_record_id"] == "psp_000000000002"
    assert (row["ledger_amount_minor"], row["psp_amount_minor"], row["diff_minor"]) == (
        None,
        10_000,
        10_000,
    )
    assert row["settlement_date"] == date(2026, 3, 20)


def test_duplicates_keep_the_lowest_psp_record_id_as_the_original(scenario):
    scenario.payment(1, amount=10_000)
    scenario.psp_record(9, 1)  # the duplicate: higher id, listed first on purpose
    scenario.psp_record(3, 1)  # the original: lowest id

    scenario.run("2026-03-11")

    row = only(scenario.breaks())
    assert row["break_type"] == BreakType.DUPLICATE_IN_PSP
    assert row["psp_record_id"] == "psp_000000000009" and row["payment_id"] == "pay_000000000001"
    assert (row["ledger_amount_minor"], row["psp_amount_minor"], row["diff_minor"]) == (
        None,
        10_000,
        10_000,
    )
    assert scenario.mart("fee_summary")["gross_minor"].to_list() == [10_000], "counted once"


def test_a_duplicate_does_not_hide_a_problem_with_the_original(scenario):
    scenario.payment(1, amount=10_000)
    scenario.psp_record(1, 1, gross=9_000, fee=90)
    scenario.psp_record(2, 1, gross=9_000, fee=90)

    scenario.run("2026-03-11")

    kinds = sorted(scenario.breaks()["break_type"].to_list())
    assert kinds == [BreakType.AMOUNT_MISMATCH, BreakType.DUPLICATE_IN_PSP]


def test_three_copies_give_two_duplicates(scenario):
    scenario.payment(1)
    for n in (1, 2, 3):
        scenario.psp_record(n, 1)

    scenario.run("2026-03-11")

    breaks = scenario.breaks()
    assert breaks["break_type"].to_list() == [BreakType.DUPLICATE_IN_PSP] * 2
    assert breaks["psp_record_id"].to_list() == ["psp_000000000002", "psp_000000000003"]


# --- refunds and chargebacks ------------------------------------------------------------------


def test_refunds_and_chargebacks_that_match_are_not_breaks(scenario):
    scenario.payment(1, amount=10_000)
    scenario.payment(2, amount=8_000)
    scenario.refund(1, payment=1, amount=2_500)  # created on 03-10, settles on 03-11
    scenario.chargeback(1, payment=2, amount=8_000)
    scenario.psp_record(1, 1)
    scenario.psp_record(2, 2, gross=8_000)
    scenario.psp_record(3, 1, record_type="REFUND", gross=2_500)
    scenario.psp_record(4, 2, record_type="CHARGEBACK", gross=8_000)

    scenario.run("2026-03-11")

    assert scenario.breaks().height == 0


def test_a_refund_the_psp_did_not_settle_is_missing_in_psp(scenario):
    scenario.payment(1)
    scenario.refund(1, payment=1, amount=2_500)
    scenario.psp_record(1, 1)

    scenario.run("2026-03-11")

    row = only(scenario.breaks())
    assert row["break_type"] == BreakType.MISSING_IN_PSP and row["ledger_amount_minor"] == 2_500


def test_a_psp_refund_the_ledger_never_recorded_is_missing_in_ledger(scenario):
    scenario.payment(1)
    scenario.psp_record(1, 1)
    scenario.psp_record(2, 1, record_type="REFUND", gross=700)

    scenario.run("2026-03-11")

    row = only(scenario.breaks())
    assert row["break_type"] == BreakType.MISSING_IN_LEDGER
    assert row["payment_id"] == "pay_000000000001", "the reference names a real payment"


def test_a_refund_with_the_wrong_amount_is_an_amount_mismatch(scenario):
    scenario.payment(1)
    scenario.refund(1, payment=1, amount=2_500)
    scenario.psp_record(1, 1)
    scenario.psp_record(2, 1, record_type="REFUND", gross=2_000)

    scenario.run("2026-03-11")

    row = only(scenario.breaks())
    assert row["break_type"] == BreakType.AMOUNT_MISMATCH and row["diff_minor"] == -500


def test_the_fee_of_refunds_and_chargebacks_is_not_compared(scenario):
    scenario.payment(1)
    scenario.chargeback(1, payment=1, amount=10_000)
    scenario.psp_record(1, 1)
    scenario.psp_record(2, 1, record_type="CHARGEBACK", gross=10_000, fee=9_999)

    scenario.run("2026-03-11")

    assert scenario.breaks().height == 0


def test_two_partial_refunds_on_the_same_day_are_not_duplicates(scenario):
    scenario.payment(1, amount=10_000)
    scenario.refund(1, payment=1, amount=2_000, on="2026-03-10")
    scenario.refund(2, payment=1, amount=3_000, on="2026-03-10")
    scenario.psp_record(1, 1)
    scenario.psp_record(2, 1, record_type="REFUND", gross=3_000)
    scenario.psp_record(3, 1, record_type="REFUND", gross=2_000)  # ids not in amount order

    scenario.run("2026-03-11")

    assert scenario.breaks().height == 0
    assert scenario.mart("daily_settlement_by_merchant")["refunds_minor"].to_list() == [5_000]


def test_a_third_psp_refund_for_two_ledger_refunds_is_a_duplicate(scenario):
    scenario.payment(1, amount=10_000)
    scenario.refund(1, payment=1, amount=2_000)
    scenario.refund(2, payment=1, amount=3_000)
    scenario.psp_record(1, 1)
    scenario.psp_record(2, 1, record_type="REFUND", gross=2_000)
    scenario.psp_record(3, 1, record_type="REFUND", gross=3_000)
    scenario.psp_record(4, 1, record_type="REFUND", gross=3_000)

    scenario.run("2026-03-11")

    row = only(scenario.breaks())
    assert row["break_type"] == BreakType.DUPLICATE_IN_PSP and row["psp_record_id"].endswith("4")


# --- what is not a break ----------------------------------------------------------------------


def test_a_payment_that_is_not_due_yet_is_not_a_break(scenario):
    scenario.merchant("mer_00000b", mdr_bps=100, settlement_days=30)
    scenario.payment(1, merchant="mer_00000b")  # captured 03-10, due 04-09

    scenario.run("2026-03-11", "2026-03-12", "2026-04-08")

    assert scenario.breaks().height == 0
    assert scenario.mart("daily_settlement_by_merchant").height == 0


def test_the_same_payment_is_missing_once_it_is_due(scenario):
    scenario.merchant("mer_00000b", mdr_bps=100, settlement_days=30)
    scenario.payment(1, merchant="mer_00000b")

    scenario.run("2026-04-09")

    row = only(scenario.breaks())
    assert row["break_type"] == BreakType.MISSING_IN_PSP and row["settlement_date"] == date(
        2026, 4, 9
    )


def test_a_voided_payment_is_never_expected(scenario):
    scenario.payment(1, status="VOIDED")

    scenario.run("2026-03-11")

    assert scenario.breaks().height == 0
    assert scenario.mart("daily_settlement_by_merchant").height == 0


def test_the_limitation_of_a_record_settled_on_another_date(scenario):
    """Known and documented: no cross-date matching, so one real gap appears as two breaks."""
    scenario.payment(1)  # due 03-11
    scenario.psp_record(1, 1, on="2026-03-12")  # the PSP settled a day late

    scenario.run("2026-03-11", "2026-03-12")

    breaks = scenario.breaks().sort("settlement_date")
    assert breaks["break_type"].to_list() == [
        BreakType.MISSING_IN_PSP,
        BreakType.MISSING_IN_LEDGER,
    ]
    assert breaks["payment_id"].to_list() == ["pay_000000000001"] * 2


# --- completeness -----------------------------------------------------------------------------


def test_every_break_type_can_appear_together_and_all_are_classified(scenario):
    scenario.payment(1)  # clean
    scenario.payment(2)  # amount mismatch
    scenario.payment(3)  # fee mismatch
    scenario.payment(4)  # missing in PSP
    scenario.payment(5)  # duplicated
    scenario.psp_record(1, 1)
    scenario.psp_record(2, 2, gross=9_500, fee=95)
    scenario.psp_record(3, 3, fee=500)
    scenario.psp_record(5, 5)
    scenario.psp_record(6, 5)
    scenario.psp_record(7, None)  # missing in ledger
    scenario.psp_record(8, 1, record_type="CHARGEBACK", on="2026-03-11")  # chargeback not in ledger

    scenario.run("2026-03-11")

    assert sorted(scenario.breaks()["break_type"].to_list()) == sorted(
        [
            BreakType.AMOUNT_MISMATCH,
            BreakType.FEE_MISMATCH,
            BreakType.MISSING_IN_PSP,
            BreakType.DUPLICATE_IN_PSP,
            BreakType.MISSING_IN_LEDGER,
            BreakType.CHARGEBACK_NOT_IN_LEDGER,
        ]
    )
