import math
from datetime import UTC, date, datetime, timedelta, timezone
from fractions import Fraction

import pytest

from payments_shared.data_contracts import load_contract
from settlement.domain import (
    CHARGEBACK_FEE_MINOR,
    EVENT_SETTLEMENT_DAYS,
    BreakType,
    Currency,
    RecordType,
    RunStatus,
    expected_fee_minor,
    settlement_date,
)
from settlement.domain.runs import (
    ALLOWED_TRANSITIONS,
    MAX_ERROR_LENGTH,
    IllegalTransitionError,
    check_transition,
    truncate_error,
)


@pytest.mark.parametrize(
    "gross, bps, expected",
    [
        (1050, 100, 11),  # exactly 10.5: half rounds up
        (1049, 100, 10),  # 10.49
        (1051, 100, 11),  # 10.51
        (1000, 100, 10),  # exact
        (50, 50, 0),  # 0.25
        (100, 50, 1),  # 0.5: half rounds up
        (99, 50, 0),  # 0.495
        (10_000, 250, 250),
        (1, 500, 0),  # 0.05
        (5_000_000, 500, 250_000),
        (0, 300, 0),
    ],
)
def test_fee_is_rounded_half_up(gross, bps, expected):
    assert expected_fee_minor(gross, bps) == expected


def test_fee_equals_exact_rational_arithmetic():
    """Check against exact fractions (no floats): floor(gross * bps / 10000 + 1/2)."""
    for gross in range(0, 3_000, 7):
        for bps in (50, 125, 250, 375, 500):
            exact = math.floor(Fraction(gross * bps, 10_000) + Fraction(1, 2))
            assert expected_fee_minor(gross, bps) == exact


@pytest.mark.parametrize("gross, bps", [(-1, 100), (100, -1)])
def test_fee_rejects_negative_inputs(gross, bps):
    with pytest.raises(ValueError):
        expected_fee_minor(gross, bps)


@pytest.mark.parametrize(
    "captured, days, expected",
    [
        (datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC), 1, date(2026, 1, 2)),
        (datetime(2026, 1, 1, 23, 59, 59, 999_999, tzinfo=UTC), 1, date(2026, 1, 2)),
        (datetime(2026, 1, 1, 12, tzinfo=UTC), 2, date(2026, 1, 3)),
        (datetime(2026, 1, 31, 20, tzinfo=UTC), 1, date(2026, 2, 1)),  # month end
        (datetime(2026, 12, 31, 1, tzinfo=UTC), 30, date(2027, 1, 30)),  # year end
        (datetime(2026, 1, 1, 12, tzinfo=UTC), EVENT_SETTLEMENT_DAYS, date(2026, 1, 2)),
    ],
)
def test_settlement_date_is_the_utc_date_plus_days(captured, days, expected):
    assert settlement_date(captured, days) == expected


def test_settlement_date_around_midnight_utc_depends_on_the_utc_day():
    just_before = datetime(2026, 3, 10, 23, 59, 59, tzinfo=UTC)
    just_after = just_before + timedelta(seconds=2)
    assert settlement_date(just_before, 1) == date(2026, 3, 11)
    assert settlement_date(just_after, 1) == date(2026, 3, 12)


def test_settlement_date_converts_other_offsets_to_utc_first():
    sao_paulo = timezone(timedelta(hours=-3))
    local = datetime(2026, 3, 10, 22, 30, tzinfo=sao_paulo)  # 01:30 UTC on the 11th
    assert settlement_date(local, 1) == date(2026, 3, 12)


def test_settlement_date_rejects_naive_timestamps_and_negative_days():
    with pytest.raises(ValueError, match="timezone"):
        settlement_date(datetime(2026, 1, 1), 1)
    with pytest.raises(ValueError, match="negative"):
        settlement_date(datetime(2026, 1, 1, tzinfo=UTC), -1)


def test_constants_match_the_data_contract_text():
    assert CHARGEBACK_FEE_MINOR == 1_500
    rule = next(
        r for r in load_contract("psp_settlement").rules if r.name == "chargeback_semantics"
    )
    assert "1500" in rule.expr


# --- enums against the data contracts ---------------------------------------------------------


def test_enums_equal_the_contract_accepted_values():
    psp = load_contract("psp_settlement")
    assert {r.value for r in RecordType} == set(
        psp.column("record_type").constraints.accepted_values
    )
    for name in ("injected_breaks", "reconciliation_breaks"):
        accepted = load_contract(name).column("break_type").constraints.accepted_values
        assert {b.value for b in BreakType} == set(accepted)
    assert {c.value for c in Currency} == set(psp.column("currency").constraints.accepted_values)
    assert {s.value for s in RunStatus} == {"PENDING", "RUNNING", "DONE", "FAILED"}


# --- run lifecycle ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "current, target",
    [
        (RunStatus.PENDING, RunStatus.RUNNING),
        (RunStatus.RUNNING, RunStatus.DONE),
        (RunStatus.RUNNING, RunStatus.FAILED),
        (RunStatus.RUNNING, RunStatus.PENDING),  # stale recovery
    ],
)
def test_legal_transitions(current, target):
    check_transition(current, target)


@pytest.mark.parametrize(
    "current, target",
    [
        (RunStatus.PENDING, RunStatus.DONE),
        (RunStatus.PENDING, RunStatus.FAILED),
        (RunStatus.PENDING, RunStatus.PENDING),
        (RunStatus.DONE, RunStatus.RUNNING),
        (RunStatus.DONE, RunStatus.PENDING),
        (RunStatus.FAILED, RunStatus.PENDING),
        (RunStatus.FAILED, RunStatus.RUNNING),
    ],
)
def test_illegal_transitions(current, target):
    with pytest.raises(IllegalTransitionError, match=f"{current} to {target}"):
        check_transition(current, target)


def test_finished_runs_are_terminal():
    assert not ALLOWED_TRANSITIONS[RunStatus.DONE] and not ALLOWED_TRANSITIONS[RunStatus.FAILED]


def test_errors_are_truncated_to_2000_characters():
    assert truncate_error("short") == "short"
    long = truncate_error("x" * 5_000)
    assert len(long) == MAX_ERROR_LENGTH and long.endswith("…")
