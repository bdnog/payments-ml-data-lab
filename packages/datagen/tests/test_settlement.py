from datetime import timedelta

import polars as pl
import pytest

from payments_datagen.generate import generate_world
from payments_datagen.profiles import SAMPLE
from payments_datagen.settlement import BREAK_TYPES, CHARGEBACK_FEE_MINOR, expected_fee_minor
from payments_shared.data_contracts import assert_valid, check_foreign_key, load_contract

SETTLEMENT = ["payments", "refunds", "chargebacks", "psp_settlement", "injected_breaks"]


@pytest.mark.parametrize("name", SETTLEMENT)
def test_every_dataset_honors_its_contract(sample_world, name):
    assert_valid(sample_world[name], load_contract(name))


def test_foreign_keys_resolve(stats_world):
    w = stats_world
    checks = [
        (w["payments"], "transaction_id", w["transactions"], "transaction_id"),
        (w["payments"], "merchant_id", w["merchants"], "merchant_id"),
        (w["payments"], "account_id", w["accounts"], "account_id"),
        (w["refunds"], "payment_id", w["payments"], "payment_id"),
        (w["chargebacks"], "payment_id", w["payments"], "payment_id"),
        (w["psp_settlement"], "merchant_id", w["merchants"], "merchant_id"),
        (w["injected_breaks"], "psp_record_id", w["psp_settlement"], "psp_record_id"),
        (w["injected_breaks"], "payment_id", w["payments"], "payment_id"),
    ]
    for child, child_col, parent, parent_col in checks:
        assert check_foreign_key(child, child_col, parent, parent_col) == []


# --- payments ---------------------------------------------------------------------------------


def test_payments_exist_only_for_approved_transactions(stats_world):
    w = stats_world
    declined = w.outcomes.filter(pl.col("declined"))["transaction_id"]
    assert w["payments"].height == w["transactions"].height - declined.len()
    assert not w["payments"]["transaction_id"].is_in(declined.implode()).any()
    assert w["payments"]["transaction_id"].n_unique() == w["payments"].height


def test_payments_copy_the_transaction_and_ids_grow_with_time(stats_world):
    w = stats_world
    joined = w["payments"].join(w["transactions"], on="transaction_id", suffix="_txn")
    assert (joined["amount_minor"] == joined["amount_minor_txn"]).all()
    assert (joined["authorized_at"] == joined["occurred_at"]).all()
    assert (joined["merchant_id"] == joined["merchant_id_txn"]).all()
    assert w["payments"]["payment_id"].is_sorted()
    assert w["payments"]["authorized_at"].is_sorted()


def test_about_3_percent_of_payments_are_voided_and_capture_takes_up_to_two_days(stats_world):
    payments = stats_world["payments"]
    assert (payments["status"] == "VOIDED").mean() == pytest.approx(0.03, abs=0.01)
    captured = payments.filter(pl.col("status") == "CAPTURED")
    delay = captured["captured_at"] - captured["authorized_at"]
    assert delay.min() >= timedelta(0)
    assert delay.max() <= timedelta(days=2)


# --- chargebacks ------------------------------------------------------------------------------


def test_fraud_chargebacks_match_the_transaction_labels(stats_world):
    w = stats_world
    fraud_cb = w["chargebacks"].filter(pl.col("reason_code") == "FRAUD")
    labeled = w["transactions"].filter(pl.col("chargeback_reported_at").is_not_null())
    joined = fraud_cb.join(w["payments"], on="payment_id", suffix="_pay").join(
        labeled, on="transaction_id"
    )
    assert fraud_cb.height == labeled.height == joined.height > 0
    assert (joined["opened_at"] == joined["chargeback_reported_at"]).all()
    assert (joined["amount_minor"] == joined["amount_minor_pay"]).all()


def test_friendly_fraud_chargebacks_hit_legitimate_payments(stats_world):
    w = stats_world
    friendly = w["chargebacks"].filter(pl.col("reason_code") != "FRAUD")
    assert friendly.height > 0
    payments = w["payments"].select("payment_id", "transaction_id", "status", "captured_at")
    labels = w["transactions"].select("transaction_id", "is_fraud")
    joined = friendly.join(payments, on="payment_id").join(labels, on="transaction_id")
    assert not joined["is_fraud"].any()
    assert (joined["status"] == "CAPTURED").all()
    assert (joined["opened_at"] > joined["captured_at"]).all()


# --- refunds ----------------------------------------------------------------------------------


def test_refunds_stay_within_the_payment_amount_and_follow_capture(stats_world):
    w = stats_world
    refunds = w["refunds"].join(w["payments"], on="payment_id", suffix="_pay")
    assert (refunds["amount_minor"] > 0).all()
    assert (refunds["amount_minor"] <= refunds["amount_minor_pay"]).all()
    assert (refunds["created_at"] > refunds["captured_at"]).all()
    totals = refunds.group_by("payment_id").agg(
        pl.col("amount_minor").sum().alias("refunded"),
        pl.col("amount_minor_pay").first().alias("paid"),
    )
    assert (totals["refunded"] <= totals["paid"]).all()


def test_refund_mix_has_partial_and_multiple_refunds_and_about_4_percent_of_payments(stats_world):
    w = stats_world
    refunds = w["refunds"].join(w["payments"], on="payment_id", suffix="_pay")
    assert (refunds["amount_minor"] < refunds["amount_minor_pay"]).any()
    assert (refunds["amount_minor"] == refunds["amount_minor_pay"]).any()
    assert w["refunds"].group_by("payment_id").len()["len"].max() == 2
    captured = w["payments"].filter(pl.col("status") == "CAPTURED").height
    assert w["refunds"]["payment_id"].n_unique() / captured == pytest.approx(0.04, abs=0.01)


def test_refunded_payments_stay_captured_and_are_never_charged_back_too(stats_world):
    w = stats_world
    refunded = w["payments"].filter(
        pl.col("payment_id").is_in(w["refunds"]["payment_id"].implode())
    )
    assert (refunded["status"] == "CAPTURED").all()
    assert not w["refunds"]["payment_id"].is_in(w["chargebacks"]["payment_id"].implode()).any()


# --- PSP file ---------------------------------------------------------------------------------


def test_expected_fee_rounds_half_up_in_integer_arithmetic():
    cases = [(1_000, 150, 15), (333, 150, 5), (50, 50, 0), (1_100, 50, 6), (1_090, 50, 5)]
    frame = pl.DataFrame({"gross": [c[0] for c in cases], "mdr": [c[1] for c in cases]})
    result = frame.select(expected_fee_minor(pl.col("gross"), pl.col("mdr")).alias("fee"))["fee"]
    assert result.to_list() == [c[2] for c in cases]
    assert result.dtype == pl.Int64


def test_without_breaks_sales_carry_exactly_the_contracted_fee(stats_world_without_breaks):
    w = stats_world_without_breaks
    sales = w["psp_settlement"].filter(pl.col("record_type") == "SALE")
    joined = sales.join(w["merchants"], on="merchant_id")
    fee = joined.select(expected_fee_minor(pl.col("gross_amount_minor"), pl.col("mdr_bps")))
    assert (joined["fee_minor"] == fee.to_series()).all()


def test_sign_semantics_of_each_record_type(stats_world):
    psp = stats_world["psp_settlement"]
    sale = psp.filter(pl.col("record_type") == "SALE")
    refund = psp.filter(pl.col("record_type") == "REFUND")
    chargeback = psp.filter(pl.col("record_type") == "CHARGEBACK")
    assert min(sale.height, refund.height, chargeback.height) > 0
    assert (sale["net_amount_minor"] == sale["gross_amount_minor"] - sale["fee_minor"]).all()
    assert (refund["fee_minor"] == 0).all()
    assert (refund["net_amount_minor"] == -refund["gross_amount_minor"]).all()
    assert (chargeback["fee_minor"] == CHARGEBACK_FEE_MINOR).all()
    assert (
        chargeback["net_amount_minor"] == -(chargeback["gross_amount_minor"] + CHARGEBACK_FEE_MINOR)
    ).all()
    assert (psp["gross_amount_minor"] > 0).all()


def test_due_date_cutoff_only_due_payments_have_sale_records(stats_world_without_breaks):
    w = stats_world_without_breaks
    last_day = w.universe.profile.last_day
    psp = w["psp_settlement"]
    assert (psp["settlement_date"] <= last_day).all()

    due = (
        w["payments"]
        .filter(pl.col("status") == "CAPTURED")
        .join(w["merchants"].select("merchant_id", "settlement_days"), on="merchant_id")
        .with_columns(
            due_date=(
                pl.col("captured_at").dt.date().cast(pl.Datetime("us"))
                + pl.duration(days=pl.col("settlement_days"))
            ).cast(pl.Date)
        )
    )
    expected_due = due.filter(pl.col("due_date") <= last_day)["payment_id"]
    not_due = due.filter(pl.col("due_date") > last_day)["payment_id"]
    sale_refs = psp.filter(pl.col("record_type") == "SALE")["merchant_reference"]

    assert not_due.len() > 0, "the profile must include payments that are not due yet"
    assert sorted(sale_refs.to_list()) == sorted(expected_due.to_list())
    assert not sale_refs.is_in(not_due.implode()).any()


def test_psp_file_is_sorted_by_settlement_date_with_ids_growing_with_it(stats_world):
    psp = stats_world["psp_settlement"]
    assert psp["settlement_date"].is_sorted()
    assert psp["psp_record_id"].is_sorted()


# --- injected breaks --------------------------------------------------------------------------


def test_break_rate_is_close_to_the_profile_and_every_type_is_present(stats_world):
    w = stats_world
    captured = w["payments"].filter(pl.col("status") == "CAPTURED").height
    breaks = w["injected_breaks"]
    assert breaks.height / captured == pytest.approx(w.universe.profile.break_rate, rel=0.1)
    assert set(breaks["break_type"]) == set(BREAK_TYPES)


def test_every_type_is_present_in_the_committed_sample_profile(sample_world):
    assert set(sample_world["injected_breaks"]["break_type"]) == set(BREAK_TYPES)


def test_at_most_one_break_per_payment_and_per_psp_record(stats_world):
    breaks = stats_world["injected_breaks"]
    with_payment = breaks.drop_nulls("payment_id")
    assert with_payment["payment_id"].n_unique() == with_payment.height
    with_record = breaks.drop_nulls("psp_record_id")
    assert with_record["psp_record_id"].n_unique() == with_record.height


def test_missing_in_psp_breaks_have_no_record_and_missing_in_ledger_have_no_payment(stats_world):
    w = stats_world
    breaks = w["injected_breaks"]
    missing = breaks.filter(pl.col("break_type") == "MISSING_IN_PSP")
    assert missing["psp_record_id"].null_count() == missing.height
    sale_refs = w["psp_settlement"].filter(pl.col("record_type") == "SALE")["merchant_reference"]
    assert not missing["payment_id"].is_in(sale_refs.implode()).any()

    orphans = breaks.filter(pl.col("break_type") == "MISSING_IN_LEDGER")
    assert orphans["payment_id"].null_count() == orphans.height
    records = w["psp_settlement"].filter(
        pl.col("psp_record_id").is_in(orphans["psp_record_id"].implode())
    )
    unknown = records["merchant_reference"].is_null() | ~records["merchant_reference"].is_in(
        w["payments"]["payment_id"].implode()
    )
    assert unknown.all()
    assert records["merchant_reference"].null_count() > 0
    assert records["merchant_reference"].null_count() < records.height


def test_duplicates_charge_back_without_ledger_and_fee_breaks_are_visible_in_the_file(stats_world):
    w = stats_world
    psp, breaks = w["psp_settlement"], w["injected_breaks"]

    dup = breaks.filter(pl.col("break_type") == "DUPLICATE_IN_PSP")
    sale_counts = psp.filter(pl.col("record_type") == "SALE").group_by("merchant_reference").len()
    twice = sale_counts.filter(pl.col("len") == 2)["merchant_reference"]
    assert dup["payment_id"].is_in(twice.implode()).all()

    cb = breaks.filter(pl.col("break_type") == "CHARGEBACK_NOT_IN_LEDGER")
    assert not cb["payment_id"].is_in(w["chargebacks"]["payment_id"].implode()).any()
    records = psp.filter(pl.col("psp_record_id").is_in(cb["psp_record_id"].implode()))
    assert (records["record_type"] == "CHARGEBACK").all()

    fee = breaks.filter(pl.col("break_type") == "FEE_MISMATCH")
    assert (fee["expected_diff_minor"] != 0).all()


def test_generation_is_deterministic():
    first = generate_world(SAMPLE)
    again = generate_world(SAMPLE)
    for name, frame in first.frames.items():
        assert frame.equals(again[name]), name
