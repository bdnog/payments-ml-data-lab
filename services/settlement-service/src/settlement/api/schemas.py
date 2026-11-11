"""Contract of the settlement-service API v1, as Pydantic models.

The item models (``DailySettlement``, ``ReconciliationBreak``) have the same field names, units and
nullability as the mart data contracts they expose (``daily_settlement_by_merchant`` and
``reconciliation_breaks``); a test fails when they drift apart. All amounts are integers in minor
units (centavos), see ADR 0005.
"""

import datetime as dt
from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, BaseModel, Field, model_validator

MERCHANT_ID = r"^mer_[0-9a-f]{6}$"
PAYMENT_ID = r"^pay_[0-9a-f]{12}$"
PSP_RECORD_ID = r"^psp_[0-9a-f]{12}$"
BREAK_ID = r"^brk_[0-9a-f]{8}$"
RUN_ID = r"^run_[0-9a-f]{12}$"

DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 1_000


class Currency(StrEnum):
    BRL = "BRL"


class BreakType(StrEnum):
    AMOUNT_MISMATCH = "AMOUNT_MISMATCH"
    MISSING_IN_PSP = "MISSING_IN_PSP"
    MISSING_IN_LEDGER = "MISSING_IN_LEDGER"
    DUPLICATE_IN_PSP = "DUPLICATE_IN_PSP"
    FEE_MISMATCH = "FEE_MISMATCH"
    CHARGEBACK_NOT_IN_LEDGER = "CHARGEBACK_NOT_IN_LEDGER"


class Mart(StrEnum):
    """The marts the batch produces. Values are the names of their data contracts."""

    DAILY_SETTLEMENT_BY_MERCHANT = "daily_settlement_by_merchant"
    RECONCILIATION_BREAKS = "reconciliation_breaks"
    FEE_SUMMARY = "fee_summary"


class RunStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


class DailySettlement(BaseModel):
    """One row of the ``daily_settlement_by_merchant`` mart."""

    settlement_date: dt.date = Field(description="Day the money moves.", examples=["2026-01-14"])
    merchant_id: str = Field(pattern=MERCHANT_ID, examples=["mer_5d7a1e"])
    currency: Currency = Field(examples=["BRL"])
    payment_count: int = Field(ge=0, description="Number of SALE records settled.", examples=[132])
    gross_minor: int = Field(ge=0, description="Sum of SALE gross amounts.", examples=[1_250_000])
    fee_minor: int = Field(ge=0, description="Sum of SALE fees.", examples=[18_750])
    refunds_minor: int = Field(ge=0, description="Sum of REFUND gross amounts.", examples=[40_000])
    chargebacks_minor: int = Field(
        ge=0, description="Sum of CHARGEBACK gross amounts plus their flat fees.", examples=[0]
    )
    net_minor: int = Field(
        description="gross - fee - refunds - chargebacks; can be negative.", examples=[1_191_250]
    )


class ReconciliationBreak(BaseModel):
    """One row of the ``reconciliation_breaks`` mart."""

    break_id: str = Field(pattern=BREAK_ID, examples=["brk_00a1b2c3"])
    run_date: dt.date = Field(
        description="Business date of the run that found it.", examples=["2026-01-15"]
    )
    settlement_date: dt.date | None = Field(
        description="Settlement date of the item; null when it cannot be determined.",
        examples=["2026-01-14"],
    )
    break_type: BreakType = Field(examples=["AMOUNT_MISMATCH"])
    payment_id: str | None = Field(pattern=PAYMENT_ID, examples=["pay_5d7a1e3f7c0b"])
    psp_record_id: str | None = Field(pattern=PSP_RECORD_ID, examples=["psp_0010db36de61"])
    ledger_amount_minor: int | None = Field(
        description="Amount on the ledger side; null when the ledger has no entry.",
        examples=[48_900],
    )
    psp_amount_minor: int | None = Field(
        description="Amount on the PSP side; null when the PSP has no record.", examples=[47_400]
    )
    diff_minor: int = Field(description="PSP minus ledger, in minor units.", examples=[-1_500])


class SettlementsPage(BaseModel):
    items: list[DailySettlement]
    next_cursor: str | None = Field(
        description="Opaque cursor for the next page; null on the last page.",
        examples=["eyJzIjoiMjAyNi0wMS0xNCJ9"],
    )


class BreaksPage(BaseModel):
    items: list[ReconciliationBreak]
    next_cursor: str | None = Field(
        description="Opaque cursor for the next page; null on the last page.",
        examples=["eyJiIjoiYnJrXzAwYTFiMmMzIn0"],
    )


class RunRequest(BaseModel):
    run_date: dt.date = Field(description="Business date to settle.", examples=["2026-01-15"])


class Run(BaseModel):
    run_id: str = Field(pattern=RUN_ID, examples=["run_4f9c2a7d10be"])
    run_date: dt.date = Field(examples=["2026-01-15"])
    status: RunStatus = Field(examples=["DONE"])
    requested_at: AwareDatetime = Field(examples=["2026-01-16T02:00:00Z"])
    started_at: AwareDatetime | None = Field(examples=["2026-01-16T02:00:03Z"])
    finished_at: AwareDatetime | None = Field(examples=["2026-01-16T02:01:12Z"])
    error: str | None = Field(description="Why the run failed; only when FAILED.", examples=[None])

    @model_validator(mode="after")
    def _lifecycle_is_consistent(self) -> Self:
        finished = self.status in (RunStatus.DONE, RunStatus.FAILED)
        if finished != (self.finished_at is not None):
            raise ValueError("finished_at is set if and only if the run is DONE or FAILED")
        if self.status == RunStatus.PENDING and self.started_at is not None:
            raise ValueError("a PENDING run has not started")
        if (self.status == RunStatus.FAILED) != (self.error is not None):
            raise ValueError("error is set if and only if the run FAILED")
        return self
