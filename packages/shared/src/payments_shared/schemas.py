"""Schemas shared across services."""

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Payload returned by every service's ``GET /health`` endpoint."""

    status: str
    service: str
    version: str


class ProblemDetails(BaseModel):
    """An RFC 7807 problem, served as ``application/problem+json``.

    Used for every error response of the HTTP services, including the ``501`` of routes that are
    part of a contract but not implemented yet.
    """

    type: str = Field(
        default="about:blank",
        description=(
            "URI that identifies the kind of problem, always `urn:problem:<slug>` here; "
            "'about:blank' means 'see status'."
        ),
        examples=["urn:problem:validation-error"],
    )
    title: str = Field(description="Short, human-readable summary.", examples=["Not Implemented"])
    status: int = Field(ge=100, le=599, description="HTTP status code.", examples=[501])
    detail: str | None = Field(
        default=None,
        description="Explanation specific to this occurrence.",
        examples=["POST /v1/score is part of the v1 contract but is not implemented yet."],
    )
    instance: str | None = Field(
        default=None, description="URI reference of this occurrence.", examples=["/v1/score"]
    )
    errors: list[dict[str, Any]] | None = Field(
        default=None,
        description="Validation issues (field location, message, type); only on 422.",
    )
