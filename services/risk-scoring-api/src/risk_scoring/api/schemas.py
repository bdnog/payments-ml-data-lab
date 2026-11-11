"""Contract of the risk-scoring API v1, as Pydantic models.

The request mirrors the *feature* columns of the ``transactions`` data contract, grouped into
nested blocks. v1 expects the caller to send the ``behavior`` block (velocity features) already
computed; computing it server-side from history is a v2 concern (ADR 0006).
``FEATURE_PATHS`` records which request field carries which contract column, and a test fails if
the two ever drift apart.
"""

from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, BaseModel, Field, model_validator

from payments_shared.schemas import ProblemDetails

MERCHANT_ID = r"^mer_[0-9a-f]{6}$"
ACCOUNT_ID = r"^acc_[0-9a-f]{8}$"
DEVICE_ID = r"^dev_[0-9a-f]{10}$"
TRANSACTION_ID = r"^txn_[0-9a-f]{12}$"
COUNTRY = r"^[A-Z]{2}$"


class Channel(StrEnum):
    WEB = "web"
    APP = "app"
    POS = "pos"


class Currency(StrEnum):
    BRL = "BRL"


class Decision(StrEnum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    DECLINE = "DECLINE"


class ReasonCode(StrEnum):
    """Positive reasons only: why the score is *high*. There are no codes for "looked fine"."""

    NEW_DEVICE = "NEW_DEVICE"
    PROXY_DETECTED = "PROXY_DETECTED"
    COUNTRY_MISMATCH = "COUNTRY_MISMATCH"
    AMOUNT_ABOVE_NORMAL = "AMOUNT_ABOVE_NORMAL"
    HIGH_VELOCITY_1H = "HIGH_VELOCITY_1H"
    HIGH_VELOCITY_24H = "HIGH_VELOCITY_24H"
    NEW_ACCOUNT = "NEW_ACCOUNT"
    HIGH_RISK_MCC = "HIGH_RISK_MCC"
    NEW_CARD_FOR_MERCHANT = "NEW_CARD_FOR_MERCHANT"


class MerchantBlock(BaseModel):
    id: str = Field(pattern=MERCHANT_ID, examples=["mer_5d7a1e"])
    mcc: str = Field(
        pattern=r"^[0-9]{4}$", description="Merchant category code.", examples=["5732"]
    )


class CardBlock(BaseModel):
    bin: str = Field(
        pattern=r"^[0-9]{6}$", description="First 6 digits of the card number.", examples=["912345"]
    )
    issuer_country: str = Field(pattern=COUNTRY, examples=["BR"])
    is_new_for_merchant: bool = Field(
        description="True when this account never used this card at this merchant before.",
        examples=[True],
    )


class DeviceBlock(BaseModel):
    id: str = Field(
        pattern=DEVICE_ID,
        description="Device fingerprint; for in-person purchases, the terminal id.",
        examples=["dev_0a1b2c3d4e"],
    )
    ip_country: str | None = Field(
        pattern=COUNTRY,
        description="Country of the client IP; null for in-person (pos) purchases.",
        examples=["RU"],
    )
    is_proxy: bool = Field(
        description="Whether the client IP looks like a proxy or VPN.", examples=[True]
    )
    is_new_for_account: bool = Field(
        description="True when this account never used this device before.", examples=[True]
    )


class AccountBlock(BaseModel):
    id: str = Field(pattern=ACCOUNT_ID, examples=["acc_33ba0290"])
    age_days: int = Field(
        ge=0, description="Whole days since the account was opened.", examples=[412]
    )


class BehaviorBlock(BaseModel):
    """Velocity features, computed point-in-time by the caller: past data only."""

    txn_count_1h: int = Field(
        ge=0, description="Account transactions in the last hour.", examples=[3]
    )
    txn_count_24h: int = Field(
        ge=0, description="Account transactions in the last 24 hours.", examples=[5]
    )
    avg_amount_30d_minor: int | None = Field(
        ge=0,
        description="Mean amount (minor units) over the last 30 days; null if there is no history.",
        examples=[18_500],
    )
    seconds_since_last_txn: int | None = Field(
        ge=0,
        description="Seconds since the account's previous transaction; null for the first.",
        examples=[240],
    )

    @model_validator(mode="after")
    def _velocity_is_monotonic(self) -> Self:
        if self.txn_count_1h > self.txn_count_24h:
            raise ValueError("txn_count_1h cannot exceed txn_count_24h")
        return self


class ScoreRequest(BaseModel):
    transaction_id: str = Field(pattern=TRANSACTION_ID, examples=["txn_0a1b2c3d4e5f"])
    occurred_at: AwareDatetime = Field(
        description="When the transaction was attempted; must carry a timezone.",
        examples=["2026-01-12T22:41:07.512345Z"],
    )
    amount_minor: int = Field(
        gt=0, description="Amount in minor units (centavos), never a float.", examples=[89_900]
    )
    currency: Currency = Field(examples=["BRL"])
    channel: Channel = Field(examples=["web"])
    installments: int = Field(ge=1, le=12, examples=[1])
    merchant: MerchantBlock
    card: CardBlock
    device: DeviceBlock
    account: AccountBlock
    behavior: BehaviorBlock


class ScoreResponse(BaseModel):
    transaction_id: str = Field(pattern=TRANSACTION_ID, examples=["txn_0a1b2c3d4e5f"])
    score: float = Field(ge=0, le=1, description="Fraud probability-like score.", examples=[0.93])
    decision: Decision = Field(examples=["DECLINE"])
    model_version: str = Field(examples=["lgbm-baseline-0.1.0"])
    reason_codes: list[ReasonCode] = Field(
        description="Why the score is high. Empty for clearly legitimate transactions.",
        examples=[["NEW_DEVICE", "PROXY_DETECTED", "AMOUNT_ABOVE_NORMAL"]],
    )
    latency_ms: float = Field(ge=0, description="Server-side scoring time.", examples=[4.2])


MAX_BATCH_SIZE = 100


class BatchScoreRequest(BaseModel):
    items: list[ScoreRequest] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class BatchScoreItemResult(BaseModel):
    """The outcome for one item: exactly one of ``result`` and ``error`` is set."""

    transaction_id: str = Field(pattern=TRANSACTION_ID, examples=["txn_0a1b2c3d4e5f"])
    result: ScoreResponse | None = None
    error: ProblemDetails | None = None

    @model_validator(mode="after")
    def _exactly_one_outcome(self) -> Self:
        if (self.result is None) == (self.error is None):
            raise ValueError("exactly one of result and error must be set")
        return self


class BatchScoreResponse(BaseModel):
    results: list[BatchScoreItemResult] = Field(
        description="One entry per request item, in the same order."
    )


class Thresholds(BaseModel):
    review: float = Field(
        ge=0, le=1, description="Scores at or above this go to review.", examples=[0.5]
    )
    decline: float = Field(
        ge=0, le=1, description="Scores at or above this are declined.", examples=[0.85]
    )

    @model_validator(mode="after")
    def _review_below_decline(self) -> Self:
        if self.review > self.decline:
            raise ValueError("review threshold cannot exceed decline threshold")
        return self


class ModelInfo(BaseModel):
    model_version: str = Field(examples=["lgbm-baseline-0.1.0"])
    trained_at: AwareDatetime = Field(examples=["2026-02-01T03:00:00Z"])
    thresholds: Thresholds
    metrics: dict[str, float] = Field(
        description="Free-form numeric evaluation metrics.",
        examples=[{"pr_auc": 0.41, "recall_at_1pct_fpr": 0.62}],
    )


# --- how the request maps to the ``transactions`` data contract ---------------------------------

# Every ``role: feature`` column of the contract and the request field that carries it.
FEATURE_PATHS: dict[str, str] = {
    "amount_minor": "amount_minor",
    "mcc": "merchant.mcc",
    "channel": "channel",
    "installments": "installments",
    "card_bin": "card.bin",
    "issuer_country": "card.issuer_country",
    "ip_country": "device.ip_country",
    "is_proxy": "device.is_proxy",
    "device_is_new_for_account": "device.is_new_for_account",
    "card_is_new_for_merchant": "card.is_new_for_merchant",
    "account_age_days": "account.age_days",
    "txn_count_1h": "behavior.txn_count_1h",
    "txn_count_24h": "behavior.txn_count_24h",
    "avg_amount_30d_minor": "behavior.avg_amount_30d_minor",
    "seconds_since_last_txn": "behavior.seconds_since_last_txn",
}

# Identity and metadata columns the request also carries (not model inputs).
IDENTITY_PATHS: dict[str, str] = {
    "transaction_id": "transaction_id",
    "occurred_at": "occurred_at",
    "currency": "currency",
    "merchant_id": "merchant.id",
    "account_id": "account.id",
    "device_id": "device.id",
}
