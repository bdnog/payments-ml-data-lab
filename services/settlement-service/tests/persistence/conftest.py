"""Fixtures for the run-repository contract tests: the same tests run on every adapter."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text

from settlement.persistence.repository import (
    InMemoryRunRepository,
    RunRepository,
    SqlRunRepository,
)
from settlement.persistence.session import session_factory

TEST_DATABASE_URL = os.environ.get("SETTLEMENT_TEST_DATABASE_URL")
SERVICE_DIR = Path(__file__).resolve().parents[2]
SKIP_SQL = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set SETTLEMENT_TEST_DATABASE_URL to run the Postgres tests (`make test-db` does it)",
)


def alembic_config(url: str) -> Config:
    config = Config(str(SERVICE_DIR / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return config


@pytest.fixture(scope="session")
def sql_engine() -> Iterator[Engine]:
    assert TEST_DATABASE_URL
    config = alembic_config(TEST_DATABASE_URL)
    command.downgrade(config, "base")
    command.upgrade(config, "head")  # the tests exercise the migration, not create_all
    engine = create_engine(TEST_DATABASE_URL, pool_size=20)
    yield engine
    engine.dispose()
    command.downgrade(config, "base")


@pytest.fixture
def clean_table(sql_engine: Engine) -> Engine:
    with sql_engine.begin() as connection:
        connection.execute(text("TRUNCATE runs"))
    return sql_engine


@pytest.fixture(params=["memory", pytest.param("sql", marks=SKIP_SQL)])
def repository(request: pytest.FixtureRequest) -> RunRepository:
    if request.param == "memory":
        return InMemoryRunRepository()
    engine: Engine = request.getfixturevalue("clean_table")
    return SqlRunRepository(session_factory(engine))
