"""Engine and session factory. One ``Session`` per request, closed when the request ends."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from risk_scoring.persistence.repository import DecisionRepository, SqlDecisionRepository

RepositoryFactory = Callable[[], Iterator[DecisionRepository]]


def create_db_engine(database_url: str) -> Engine:
    # pool_pre_ping: a stale connection (database restarted) is replaced, not surfaced to a caller.
    return create_engine(database_url, pool_pre_ping=True)


def check_connection(engine: Engine) -> None:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))


def sql_repository_factory(engine: Engine) -> RepositoryFactory:
    """A generator that yields a repository bound to a fresh session (a FastAPI dependency body)."""
    session_factory = sessionmaker(engine, expire_on_commit=False)

    def provide() -> Iterator[DecisionRepository]:
        with session_factory() as session:
            yield SqlDecisionRepository(session)

    return provide


@contextmanager
def open_session(engine: Engine) -> Iterator[Session]:
    with sessionmaker(engine, expire_on_commit=False)() as session:
        yield session
