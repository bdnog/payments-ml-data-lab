"""Engine and session factory."""

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker


def create_db_engine(database_url: str) -> Engine:
    # pool_pre_ping: a stale connection (database restarted) is replaced, not surfaced to a caller.
    return create_engine(database_url, pool_pre_ping=True)


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


def check_connection(engine: Engine) -> None:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
