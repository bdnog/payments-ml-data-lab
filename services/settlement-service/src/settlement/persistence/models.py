"""SQLAlchemy 2.0 model of the ``runs`` table."""

from datetime import date, datetime
from typing import Any

from sqlalchemy import Date, DateTime, Index, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RunRow(Base):
    """One settlement run. At most one per ``run_date`` that is not FAILED (partial unique index).

    The first six columns are the public contract (``Run`` in the API). ``attempts``, ``claimed_at``
    and ``summary`` are internal: how often it was tried, when the current worker took it (for stale
    detection), and the numbers the pipeline reported.
    """

    __tablename__ = "runs"
    __table_args__ = (
        # "One live run per date": FAILED runs are history, anything else blocks a new request.
        Index(
            "uq_runs_run_date_not_failed",
            "run_date",
            unique=True,
            postgresql_where=text("status <> 'FAILED'"),
        ),
        # Claiming looks for the oldest PENDING run.
        Index("ix_runs_status_requested_at", "status", "requested_at"),
    )

    run_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    run_date: Mapped[date] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(16))
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)

    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
