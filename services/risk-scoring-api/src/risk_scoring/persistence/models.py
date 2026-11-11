"""SQLAlchemy 2.0 declarative model of the decision log (one row per scored transaction)."""

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, Index, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DecisionRow(Base):
    """The decision log. The primary key ``transaction_id`` is what makes scoring idempotent.

    ``request`` keeps the canonical request payload (audit: what exactly was scored), and
    ``request_hash`` its sha256, so a replay can tell "same request again" from "same id, other
    payload" without comparing JSON.
    """

    __tablename__ = "decisions"
    __table_args__ = (Index("ix_decisions_created_at", "created_at"),)

    transaction_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    request_hash: Mapped[str] = mapped_column(String(64))
    request: Mapped[dict[str, Any]] = mapped_column(JSONB)
    score: Mapped[float] = mapped_column(Float)  # FLOAT is double precision on Postgres: exact
    decision: Mapped[str] = mapped_column(String(16))
    model_version: Mapped[str] = mapped_column(String(64))
    reason_codes: Mapped[list[str]] = mapped_column(JSONB)
    latency_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
