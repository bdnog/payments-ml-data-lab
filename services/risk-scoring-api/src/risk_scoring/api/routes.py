"""Routes of the risk-scoring API v1.

Only ``/health`` (in ``main.py``) does real work. The ``/v1`` routes are the *contract*: they
declare request and response models and every status code, and answer ``501`` until phase 3.
"""

from fastapi import APIRouter, Request

from payments_shared.problem import not_implemented, problem_responses
from risk_scoring.api.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    ModelInfo,
    ScoreRequest,
    ScoreResponse,
)

router = APIRouter(prefix="/v1")

SCORE_DESCRIPTION = """\
Score one transaction.

The caller sends the transaction plus its `behavior` block (velocity features computed
point-in-time: only past data, the current transaction excluded).

**Idempotent by `transaction_id`.** Sending the same payload again returns the same decision.
Sending the same `transaction_id` with a *different* payload returns `409`.

Validation errors return `422`. All errors are `application/problem+json`.
"""


@router.post(
    "/score",
    operation_id="score",
    summary="Score one transaction",
    description=SCORE_DESCRIPTION,
    response_model=ScoreResponse,
    responses=problem_responses(409, 422, 501),
    tags=["scoring"],
)
def score(body: ScoreRequest, request: Request):
    return not_implemented(request)


@router.post(
    "/score:batch",
    operation_id="score_batch",
    summary="Score up to 100 transactions",
    description=(
        "Each item is scored independently and gets either a result or an error, in request order. "
        "A bad item does not fail the batch. The same idempotency rule as `/v1/score` applies."
    ),
    response_model=BatchScoreResponse,
    responses=problem_responses(422, 501),
    tags=["scoring"],
)
def score_batch(body: BatchScoreRequest, request: Request):
    return not_implemented(request)


@router.get(
    "/model",
    operation_id="get_model",
    summary="Describe the model in production",
    response_model=ModelInfo,
    responses=problem_responses(501),
    tags=["model"],
)
def get_model(request: Request):
    return not_implemented(request)
