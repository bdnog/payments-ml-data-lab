"""Fixtures for the repository contract tests: the same tests run on every adapter."""

import os
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import sessionmaker

from risk_scoring.persistence.repository import (
    DecisionRepository,
    InMemoryDecisionRepository,
    SqlDecisionRepository,
)

TEST_DATABASE_URL = os.environ.get("RISK_TEST_DATABASE_URL")
SERVICE_DIR = Path(__file__).resolve().parents[2]
SKIP_SQL = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set RISK_TEST_DATABASE_URL to run the Postgres tests (`make test-db` does it)",
)

# A way to open a repository "like a request would": a fresh one per call (own session for SQL).
OpenRepository = Callable[[], AbstractContextManager[DecisionRepository]]


def alembic_config(url: str) -> Config:
    config = Config(str(SERVICE_DIR / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return config


@pytest.fixture(scope="session")
def sql_engine() -> Iterator[Engine]:
    assert TEST_DATABASE_URL
    config = alembic_config(TEST_DATABASE_URL)
    command.downgrade(config, "base")  # start from nothing, even after an aborted earlier run
    command.upgrade(config, "head")  # the tests exercise the migration, not create_all
    engine = create_engine(TEST_DATABASE_URL, pool_size=20)
    yield engine
    engine.dispose()
    command.downgrade(config, "base")


@pytest.fixture
def clean_table(sql_engine: Engine) -> Engine:
    with sql_engine.begin() as connection:
        connection.execute(text("TRUNCATE decisions"))
    return sql_engine


@pytest.fixture(params=["memory", pytest.param("sql", marks=SKIP_SQL)])
def storage(request: pytest.FixtureRequest):
    """``(open_repository, count_rows)`` for each adapter."""
    if request.param == "memory":
        repository = InMemoryDecisionRepository()

        @contextmanager
        def open_memory() -> Iterator[DecisionRepository]:
            yield repository

        return open_memory, lambda: len(repository)

    engine: Engine = request.getfixturevalue("clean_table")
    sessions = sessionmaker(engine, expire_on_commit=False)

    @contextmanager
    def open_sql() -> Iterator[DecisionRepository]:
        with sessions() as session:
            yield SqlDecisionRepository(session)

    def count() -> int:
        with engine.connect() as connection:
            return connection.execute(text("SELECT count(*) FROM decisions")).scalar_one()

    return open_sql, count
