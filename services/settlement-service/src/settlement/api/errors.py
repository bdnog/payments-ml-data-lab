"""Errors of the settlement API and how they become ``application/problem+json``."""

import logging

from fastapi import FastAPI, Request
from sqlalchemy.exc import SQLAlchemyError

from payments_shared.problem import problem_response, problem_type
from settlement.storage.cursors import InvalidCursorError

log = logging.getLogger("settlement.api")


class InvalidParameterError(Exception):
    """A parameter that is well-formed but not acceptable (for example ``from`` after ``to``)."""


class PartitionNotFoundError(Exception):
    """No partition of the mart exists for the requested date: the run never happened."""


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(InvalidCursorError)
    async def _invalid_cursor(request: Request, exc: InvalidCursorError):
        return problem_response(
            request,
            400,
            "Invalid cursor",
            f"{exc}. Use the `next_cursor` of a previous page, unchanged.",
            type=problem_type("invalid-cursor"),
        )

    @app.exception_handler(InvalidParameterError)
    async def _invalid_parameter(request: Request, exc: InvalidParameterError):
        return problem_response(
            request, 422, "Validation error", str(exc), type=problem_type("validation-error")
        )

    @app.exception_handler(PartitionNotFoundError)
    async def _partition_not_found(request: Request, exc: PartitionNotFoundError):
        return problem_response(
            request, 404, "Partition not found", str(exc), type=problem_type("partition-not-found")
        )

    @app.exception_handler(SQLAlchemyError)
    async def _database(request: Request, exc: SQLAlchemyError):
        log.error("database unavailable: %s", exc.__class__.__name__, exc_info=exc)
        return problem_response(
            request,
            503,
            "Database unavailable",
            "The runs database could not be reached; retry shortly.",
            type=problem_type("database-unavailable"),
        )
