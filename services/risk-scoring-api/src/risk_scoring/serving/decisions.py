"""Score a request and record the decision, idempotently. No web framework in sight.

The flow for each request (ADR 0009):

1. score it (deterministic: the same request always gives the same score);
2. ``save_or_get``: insert the decision, or get the one that is already there;
3. if the stored ``request_hash`` equals this request's hash it is a replay (return the *stored*
   response, including its original ``latency_ms`` and ``model_version``); if not, the same
   ``transaction_id`` came with a different payload: a conflict.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from risk_scoring.api.schemas import Decision, ReasonCode, ScoreRequest, ScoreResponse
from risk_scoring.persistence.repository import (
    DecisionRecord,
    DecisionRepository,
    canonical_payload,
    hash_request,
)
from risk_scoring.serving.scorer import Scored, Scorer

log = logging.getLogger("risk_scoring.decisions")


class IdempotencyConflictError(Exception):
    """The transaction id was already scored with a different payload."""

    def __init__(self, transaction_id: str) -> None:
        super().__init__(
            f"transaction_id {transaction_id} was already scored with a different payload"
        )
        self.transaction_id = transaction_id


@dataclass(frozen=True)
class Outcome:
    """The result for one request: a response, or the conflict that prevented one."""

    response: ScoreResponse | None = None
    conflict: IdempotencyConflictError | None = None


def to_record(
    request: ScoreRequest, scored: Scored, model_version: str, latency_ms: float
) -> DecisionRecord:
    return DecisionRecord(
        transaction_id=request.transaction_id,
        request_hash=hash_request(request),
        request=canonical_payload(request),
        score=scored.score,
        decision=scored.decision.value,
        model_version=model_version,
        reason_codes=[c.value for c in scored.reason_codes],
        latency_ms=latency_ms,
    )


def to_response(record: DecisionRecord) -> ScoreResponse:
    return ScoreResponse(
        transaction_id=record.transaction_id,
        score=record.score,
        decision=Decision(record.decision),
        model_version=record.model_version,
        reason_codes=[ReasonCode(c) for c in record.reason_codes],
        latency_ms=record.latency_ms,
    )


def score_and_log_batch(
    scorer: Scorer, repository: DecisionRepository, requests: Sequence[ScoreRequest]
) -> list[Outcome]:
    """One batched forward pass, then per-item idempotency. Results are in request order."""
    timed = scorer.timed(requests)
    outcomes: list[Outcome] = []
    for request, scored in zip(requests, timed.results, strict=True):
        candidate = to_record(request, scored, scorer.model_version, timed.latency_ms_per_item)
        saved = repository.save_or_get(candidate)
        if saved.record.request_hash != candidate.request_hash:
            log.warning("transaction_id=%s outcome=conflict", request.transaction_id)
            outcomes.append(Outcome(conflict=IdempotencyConflictError(request.transaction_id)))
            continue
        stored = saved.record
        log.info(
            "transaction_id=%s decision=%s score=%.6f latency_ms=%.3f model_version=%s replay=%s",
            stored.transaction_id,
            stored.decision,
            stored.score,
            stored.latency_ms,
            stored.model_version,
            str(not saved.created).lower(),
        )
        outcomes.append(Outcome(response=to_response(stored)))
    return outcomes


def score_and_log(
    scorer: Scorer, repository: DecisionRepository, request: ScoreRequest
) -> ScoreResponse:
    outcome = score_and_log_batch(scorer, repository, [request])[0]
    if outcome.conflict is not None:
        raise outcome.conflict
    assert outcome.response is not None
    return outcome.response
