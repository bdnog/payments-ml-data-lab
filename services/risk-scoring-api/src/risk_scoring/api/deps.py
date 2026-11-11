"""FastAPI dependencies: the loaded scorer and a decision repository, from ``app.state``."""

from collections.abc import Iterator

from fastapi import Request

from risk_scoring.api.errors import ModelUnavailableError
from risk_scoring.persistence.repository import DecisionRepository
from risk_scoring.serving.scorer import Scorer


def get_scorer(request: Request) -> Scorer:
    scorer: Scorer | None = getattr(request.app.state, "scorer", None)
    if scorer is None:
        raise ModelUnavailableError
    return scorer


def get_repository(request: Request) -> Iterator[DecisionRepository]:
    """One repository (for SQL, one session) per request, closed when the request ends."""
    yield from request.app.state.repository_factory()
