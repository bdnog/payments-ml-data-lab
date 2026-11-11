"""FastAPI dependencies: the mart store and the run repository, from ``app.state``."""

from fastapi import Request

from settlement.persistence.repository import RunRepository
from settlement.storage.marts import MartStore


def get_store(request: Request) -> MartStore:
    return request.app.state.store


def get_repository(request: Request) -> RunRepository:
    return request.app.state.repository
