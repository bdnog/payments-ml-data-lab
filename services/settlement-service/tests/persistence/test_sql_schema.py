"""Postgres-only checks of the schema, the partial unique index and the migration."""

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from settlement.persistence.models import Base

from .conftest import SKIP_SQL

pytestmark = SKIP_SQL

INSERT = text(
    "INSERT INTO runs (run_id, run_date, status, requested_at) VALUES (:id, :date, :status, now())"
)


def test_migration_creates_the_documented_table(sql_engine):
    inspector = inspect(sql_engine)
    columns = [c["name"] for c in inspector.get_columns("runs")]

    assert columns == [
        "run_id", "run_date", "status", "requested_at", "started_at", "finished_at",
        "error", "attempts", "claimed_at", "summary",
    ]  # fmt: skip
    assert inspector.get_pk_constraint("runs")["constrained_columns"] == ["run_id"]
    indexes = {i["name"]: i for i in inspector.get_indexes("runs")}
    assert indexes["uq_runs_run_date_not_failed"]["unique"]
    assert "ix_runs_status_requested_at" in indexes


def test_migration_and_orm_model_agree(sql_engine):
    with sql_engine.connect() as connection:
        diff = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    assert diff == []


def test_the_database_rejects_a_second_live_run_for_a_date(clean_table):
    with clean_table.begin() as connection:
        connection.execute(
            INSERT, {"id": "run_000000000001", "date": "2026-01-15", "status": "PENDING"}
        )
    with pytest.raises(IntegrityError), clean_table.begin() as connection:
        connection.execute(
            INSERT, {"id": "run_000000000002", "date": "2026-01-15", "status": "DONE"}
        )


def test_the_database_allows_a_new_run_next_to_failed_ones(clean_table):
    with clean_table.begin() as connection:
        for n, status in enumerate(["FAILED", "FAILED", "PENDING"], start=1):
            connection.execute(
                INSERT, {"id": f"run_00000000000{n}", "date": "2026-01-15", "status": status}
            )
        count = connection.execute(text("SELECT count(*) FROM runs")).scalar_one()
    assert count == 3


def test_claim_skips_a_run_another_worker_has_locked_instead_of_waiting(clean_table):
    """What SKIP LOCKED is for: a locked row is passed over, never waited on."""
    import threading
    from datetime import UTC, date, datetime, timedelta

    from settlement.persistence.repository import SqlRunRepository
    from settlement.persistence.session import session_factory

    repository = SqlRunRepository(session_factory(clean_table))
    t0 = datetime(2026, 2, 1, tzinfo=UTC)
    oldest = repository.create(date(2026, 1, 1), now=t0).run
    second = repository.create(date(2026, 1, 2), now=t0 + timedelta(minutes=1)).run

    claimed: list = []
    with clean_table.connect() as holder:  # another worker, mid-claim: it holds the oldest row
        holder.execute(
            text("SELECT run_id FROM runs WHERE run_id = :id FOR UPDATE"), {"id": oldest.run_id}
        )
        thread = threading.Thread(target=lambda: claimed.append(repository.claim_next()))
        thread.start()
        thread.join(timeout=5)
        finished_without_waiting = not thread.is_alive()
        holder.rollback()  # release the lock (also unblocks a waiting thread in a broken build)
    thread.join(timeout=10)

    assert finished_without_waiting, "claim_next waited for a locked row"
    assert claimed[0] is not None and claimed[0].run_id == second.run_id
