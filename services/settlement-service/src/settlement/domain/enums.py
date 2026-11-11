"""The vocabulary of settlement. Values equal the ``accepted_values`` of the data contracts."""

from enum import StrEnum


class RecordType(StrEnum):
    """What a ledger or PSP record settles."""

    SALE = "SALE"
    REFUND = "REFUND"
    CHARGEBACK = "CHARGEBACK"


class BreakType(StrEnum):
    """Why a ledger item and a PSP record do not agree. There is no "unclassified"."""

    AMOUNT_MISMATCH = "AMOUNT_MISMATCH"
    MISSING_IN_PSP = "MISSING_IN_PSP"
    MISSING_IN_LEDGER = "MISSING_IN_LEDGER"
    DUPLICATE_IN_PSP = "DUPLICATE_IN_PSP"
    FEE_MISMATCH = "FEE_MISMATCH"
    CHARGEBACK_NOT_IN_LEDGER = "CHARGEBACK_NOT_IN_LEDGER"


class RunStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


class Currency(StrEnum):
    BRL = "BRL"
