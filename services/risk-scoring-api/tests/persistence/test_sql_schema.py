"""Postgres-only checks of the schema and of the migration."""

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text

from risk_scoring.persistence.models import Base

from .conftest import SKIP_SQL

pytestmark = SKIP_SQL


def test_migration_creates_the_documented_table(sql_engine):
    inspector = inspect(sql_engine)
    columns = {c["name"]: c for c in inspector.get_columns("decisions")}

    assert list(columns) == [
        "transaction_id",
        "request_hash",
        "request",
        "score",
        "decision",
        "model_version",
        "reason_codes",
        "latency_ms",
        "created_at",
    ]
    assert inspector.get_pk_constraint("decisions")["constrained_columns"] == ["transaction_id"]
    assert str(columns["request"]["type"]) == "JSONB"
    assert str(columns["reason_codes"]["type"]) == "JSONB"
    assert columns["created_at"]["type"].timezone is True
    assert columns["created_at"]["default"] is not None  # server default now()
    assert {i["name"] for i in inspector.get_indexes("decisions")} == {"ix_decisions_created_at"}


def test_migration_and_orm_model_agree(sql_engine):
    """If someone edits the model and forgets the migration, this fails."""
    with sql_engine.connect() as connection:
        diff = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    assert diff == []


def test_the_database_itself_rejects_a_duplicate_primary_key(clean_table):
    insert = text(
        "INSERT INTO decisions (transaction_id, request_hash, request, score, decision, "
        "model_version, reason_codes, latency_ms) VALUES "
        "('txn_0a1b2c3d4e5f', 'h', '{}', 0.1, 'APPROVE', 'm', '[]', 1.0)"
    )
    with clean_table.begin() as connection:
        connection.execute(insert)
    with pytest.raises(Exception, match="duplicate key"), clean_table.begin() as connection:
        connection.execute(insert)
