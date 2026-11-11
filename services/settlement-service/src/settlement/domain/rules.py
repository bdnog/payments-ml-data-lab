"""Settlement rules as plain Python: no Polars, FastAPI or SQLAlchemy in here.

These two functions are the *definition* of the rules. The batch layer has Polars expression
equivalents (``settlement.batch.expressions``) so it can apply them to millions of rows, and a
parity test proves expression and function agree on a grid of values. The data generator
(``payments_datagen.settlement``) encodes the same rules; the reconciliation is verified against it.
"""

from datetime import UTC, date, datetime, timedelta

CHARGEBACK_FEE_MINOR = 1_500  # flat fee the PSP charges per chargeback
EVENT_SETTLEMENT_DAYS = 1  # refunds and chargebacks settle the day after the event
CURRENCY = "BRL"
BASIS_POINTS = 10_000


def expected_fee_minor(gross_minor: int, mdr_bps: int) -> int:
    """Contracted SALE fee: ``round_half_up(gross * mdr_bps / 10000)``, in integer arithmetic.

    Exact halves round up: gross 1050 at 100 bps is 10.5, so 11. There are no floats anywhere; a
    float would turn a half into 10.499999 or 10.5 depending on the value.
    """
    if gross_minor < 0 or mdr_bps < 0:
        raise ValueError("gross and mdr_bps must not be negative")
    return (gross_minor * mdr_bps + BASIS_POINTS // 2) // BASIS_POINTS


def settlement_date(event_at: datetime, settlement_days: int) -> date:
    """The day the money moves: the UTC calendar date of the event plus ``settlement_days``.

    For a SALE the event is the capture and ``settlement_days`` is the merchant's (1, 2 or 30); for
    a REFUND or CHARGEBACK it is the event date and ``EVENT_SETTLEMENT_DAYS``. Only the date of the
    timestamp counts, so 23:59:59 and 00:00:01 UTC are different days.
    """
    if event_at.tzinfo is None:
        raise ValueError("event_at must be timezone-aware")
    if settlement_days < 0:
        raise ValueError("settlement_days must not be negative")
    return event_at.astimezone(UTC).date() + timedelta(days=settlement_days)
