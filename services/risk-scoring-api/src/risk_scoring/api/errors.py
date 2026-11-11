"""Errors of the risk-scoring API and how they become ``application/problem+json``."""

import logging

from fastapi import FastAPI, Request
from sqlalchemy.exc import SQLAlchemyError

from payments_shared.problem import ProblemJSONResponse, problem_response, problem_type
from risk_scoring.serving.decisions import IdempotencyConflictError

log = logging.getLogger("risk_scoring.api")


class ModelUnavailableError(Exception):
    """No model is loaded (the artifact is missing or failed verification at startup)."""


def conflict_problem(request: Request, transaction_id: str) -> ProblemJSONResponse:
    return problem_response(
        request,
        409,
        "Idempotency conflict",
        f"transaction_id {transaction_id} was already scored with a different payload.",
        type=problem_type("idempotency-conflict"),
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(IdempotencyConflictError)
    async def _conflict(request: Request, exc: IdempotencyConflictError):
        return conflict_problem(request, exc.transaction_id)

    @app.exception_handler(ModelUnavailableError)
    async def _no_model(request: Request, exc: ModelUnavailableError):
        return problem_response(
            request,
            503,
            "Model unavailable",
            "No model is loaded; see GET /ready.",
            type=problem_type("model-unavailable"),
        )

    @app.exception_handler(SQLAlchemyError)
    async def _database(request: Request, exc: SQLAlchemyError):
        # Without the decision log a decision cannot be made idempotent, so we refuse rather than
        # score without recording. The caller applies its own fallback (docs/risk-scoring.md).
        log.error("decision log unavailable: %s", exc.__class__.__name__, exc_info=exc)
        return problem_response(
            request,
            503,
            "Decision log unavailable",
            "The decision log could not be reached; retry with the same transaction_id.",
            type=problem_type("decision-log-unavailable"),
        )
