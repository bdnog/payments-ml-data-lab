"""Schemas shared across services."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Payload returned by every service's ``GET /health`` endpoint."""

    status: str
    service: str
    version: str
