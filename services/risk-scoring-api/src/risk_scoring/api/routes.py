"""Routes of the risk-scoring API v1.

Plain ``def`` handlers, not ``async def``: scoring is CPU-bound and blocking (torch, NumPy, a
synchronous database driver). FastAPI runs a ``def`` handler in a worker thread, so the event loop
stays free to accept connections. Declaring them ``async`` would run the model on the event loop and
stall every other request while it computes (ADR 0009).
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from payments_shared.problem import problem_responses, problem_type
from payments_shared.schemas import ProblemDetails
from risk_scoring.api.deps import get_repository, get_scorer
from risk_scoring.api.schemas import (
    BatchScoreItemResult,
    BatchScoreRequest,
    BatchScoreResponse,
    ModelInfo,
    RejectedItem,
    ScoreRequest,
    ScoreResponse,
    Thresholds,
)
from risk_scoring.persistence.repository import DecisionRepository
from risk_scoring.serving.decisions import score_and_log, score_and_log_batch
from risk_scoring.serving.scorer import Scorer

router = APIRouter(prefix="/v1")

ScorerDep = Annotated[Scorer, Depends(get_scorer)]
RepositoryDep = Annotated[DecisionRepository, Depends(get_repository)]

SCORE_DESCRIPTION = """\
Score one transaction.

The caller sends the transaction plus its `behavior` block (velocity features computed
point-in-time: only past data, the current transaction excluded).

`score` is a **ranking score in [0, 1], not a calibrated probability**: use `decision`, which comes
from thresholds chosen on validation data (see `GET /v1/model`). `reason_codes` are heuristic
explanations, not attributions of the model.

**Idempotent by `transaction_id`.** Sending the same payload again returns the same decision (the
stored one, including its original `latency_ms` and `model_version`). Sending the same
`transaction_id` with a *different* payload returns `409`.

Validation errors return `422`. If the model is not loaded or the decision log is down the answer
is `503`. All errors are `application/problem+json`.
"""


@router.post(
    "/score",
    operation_id="score",
    summary="Score one transaction",
    description=SCORE_DESCRIPTION,
    response_model=ScoreResponse,
    responses=problem_responses(409, 422, 503),
    tags=["scoring"],
)
def score(body: ScoreRequest, scorer: ScorerDep, repository: RepositoryDep) -> ScoreResponse:
    return score_and_log(scorer, repository, body)


@router.post(
    "/score:batch",
    operation_id="score_batch",
    summary="Score up to 100 transactions",
    description=(
        "Each item is scored independently and gets either a result or an error, in request order. "
        "A bad item does not fail the batch: an item that fails validation (but has a valid "
        "`transaction_id`) or conflicts with an earlier payload gets a per-item `error`. The same "
        "idempotency rule as `/v1/score` applies per item. The items are scored in one batched "
        "forward pass, so `latency_ms` of each result is the batch time divided by its size."
    ),
    response_model=BatchScoreResponse,
    response_model_exclude_none=True,  # an item has a result or an error, never a null of the other
    responses=problem_responses(422, 503),
    tags=["scoring"],
)
def score_batch(
    body: BatchScoreRequest, scorer: ScorerDep, repository: RepositoryDep
) -> BatchScoreResponse:
    entries = body.entries
    valid = [(i, e) for i, e in enumerate(entries) if isinstance(e, ScoreRequest)]
    outcomes = score_and_log_batch(scorer, repository, [e for _, e in valid])
    by_position = {i: outcome for (i, _), outcome in zip(valid, outcomes, strict=True)}

    results: list[BatchScoreItemResult] = []
    for position, entry in enumerate(entries):
        if isinstance(entry, RejectedItem):
            results.append(
                BatchScoreItemResult(
                    transaction_id=entry.transaction_id,
                    error=ProblemDetails(
                        type=problem_type("validation-error"),
                        title="Validation error",
                        status=422,
                        detail="This item is not a valid transaction.",
                        instance=f"/v1/score:batch#/items/{position}",
                        errors=entry.errors,
                    ),
                )
            )
            continue
        outcome = by_position[position]
        if outcome.conflict is not None:
            results.append(
                BatchScoreItemResult(
                    transaction_id=entry.transaction_id,
                    error=ProblemDetails(
                        type=problem_type("idempotency-conflict"),
                        title="Idempotency conflict",
                        status=409,
                        detail=str(outcome.conflict) + ".",
                        instance=f"/v1/score:batch#/items/{position}",
                    ),
                )
            )
        else:
            results.append(
                BatchScoreItemResult(transaction_id=entry.transaction_id, result=outcome.response)
            )
    return BatchScoreResponse(results=results)


@router.get(
    "/model",
    operation_id="get_model",
    summary="Describe the model in production",
    response_model=ModelInfo,
    responses=problem_responses(503),
    tags=["model"],
)
def get_model(scorer: ScorerDep) -> ModelInfo:
    info = scorer.info()
    return ModelInfo(
        model_version=info["model_version"],
        trained_at=info["trained_at"],
        thresholds=Thresholds(**info["thresholds"]),
        metrics=info["metrics"],
    )
