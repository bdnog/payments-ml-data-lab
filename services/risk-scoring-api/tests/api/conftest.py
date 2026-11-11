import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from payments_shared.paths import contracts_dir
from risk_scoring.api.main import create_app
from risk_scoring.config import Settings
from risk_scoring.persistence.repository import DecisionRepository, InMemoryDecisionRepository

EXAMPLES = contracts_dir() / "examples" / "risk-scoring"


@pytest.fixture(scope="session")
def request_example() -> dict[str, Any]:
    return json.loads((EXAMPLES / "score.request.json").read_text())


@pytest.fixture
def repository() -> InMemoryDecisionRepository:
    return InMemoryDecisionRepository()


@pytest.fixture
def client(model_dir, repository) -> Iterator[TestClient]:
    """The app with a small trained model and an in-memory decision log (lifespan runs)."""

    def factory() -> Iterator[DecisionRepository]:
        yield repository

    app = create_app(Settings(app_version="1.0.0", model_dir=model_dir), repository_factory=factory)
    with TestClient(app) as test_client:
        yield test_client
