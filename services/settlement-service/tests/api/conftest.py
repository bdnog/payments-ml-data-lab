from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from settlement.api.main import create_app
from settlement.batch.pipeline import run_dates
from settlement.config import Settings
from settlement.persistence.repository import InMemoryRunRepository

FIRST_DAY = date(2026, 1, 1)
DAYS = 30


@pytest.fixture(scope="session")
def marts_dir(tmp_path_factory, sample_dir) -> Path:
    """The marts of a real run over the committed sample (30 days), written once per session."""
    directory = tmp_path_factory.mktemp("marts")
    run_dates([FIRST_DAY + timedelta(days=i) for i in range(DAYS)], sample_dir, directory)
    return directory


@pytest.fixture
def repository() -> InMemoryRunRepository:
    return InMemoryRunRepository()


@pytest.fixture
def make_client(marts_dir, repository):
    def build(**overrides) -> Iterator[TestClient]:
        settings = Settings(app_version="1.0.0", marts_dir=overrides.pop("marts_dir", marts_dir))
        app = create_app(settings, repository=repository, **overrides)
        with TestClient(app) as client:
            yield client

    return build


@pytest.fixture
def client(make_client) -> Iterator[TestClient]:
    yield from make_client()
