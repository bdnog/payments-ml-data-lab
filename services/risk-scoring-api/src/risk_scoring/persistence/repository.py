"""The decision-log port and its two adapters.

``DecisionRepository`` is the *port*: what the service needs from storage, in domain terms. The
*adapters* implement it: ``InMemoryDecisionRepository`` (unit tests, no database) and
``SqlDecisionRepository`` (Postgres). The same contract tests run against both, so the in-memory
one is a trustworthy stand-in. In Spring terms this is a repository interface with two
implementations; ``Protocol`` is Python's structural version of an interface (no ``implements``).

The one operation that matters is ``save_or_get``: *insert first*, and if the transaction id was
already taken, return the row that is there. The caller decides what a pre-existing row means
(same ``request_hash``: a replay; different: a conflict). Insert-first, instead of
select-then-insert, is what makes concurrent identical requests converge without a race (ADR 0009).
"""

import hashlib
import json
import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from risk_scoring.api.schemas import ScoreRequest
from risk_scoring.persistence.models import DecisionRow


@dataclass(frozen=True)
class DecisionRecord:
    transaction_id: str
    request_hash: str
    request: dict[str, Any]
    score: float
    decision: str
    model_version: str
    reason_codes: list[str]
    latency_ms: float
    created_at: datetime | None = None


@dataclass(frozen=True)
class SaveResult:
    record: DecisionRecord  # the row that is in the log after the call (new or pre-existing)
    created: bool  # True if this call inserted it


class DecisionRepository(Protocol):
    def save_or_get(self, record: DecisionRecord) -> SaveResult:
        """Insert ``record`` unless its transaction id exists; return what is stored. Atomic."""
        ...

    def get(self, transaction_id: str) -> DecisionRecord | None: ...


def canonical_payload(request: ScoreRequest) -> dict[str, Any]:
    """The request as plain JSON values (timestamps as ISO strings): what is stored and hashed."""
    return request.model_dump(mode="json")


def hash_request(request: ScoreRequest) -> str:
    """sha256 of the canonical JSON (sorted keys, no whitespace).

    Two requests with the same content hash equal whatever order the client sent the keys in.
    A timestamp is hashed as serialized, so the same instant written with another UTC offset counts
    as a different payload: deliberately conservative for an audit log.
    """
    canonical = json.dumps(
        canonical_payload(request), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class InMemoryDecisionRepository:
    """A dict behind a lock. Same semantics as the SQL adapter, for tests without a database."""

    def __init__(self) -> None:
        self._rows: dict[str, DecisionRecord] = {}
        self._lock = threading.Lock()

    def save_or_get(self, record: DecisionRecord) -> SaveResult:
        with self._lock:
            existing = self._rows.get(record.transaction_id)
            if existing is not None:
                return SaveResult(existing, created=False)
            stored = replace(record, created_at=datetime.now(UTC))
            self._rows[record.transaction_id] = stored
            return SaveResult(stored, created=True)

    def get(self, transaction_id: str) -> DecisionRecord | None:
        with self._lock:
            return self._rows.get(transaction_id)

    def __len__(self) -> int:
        return len(self._rows)


class SqlDecisionRepository:
    """Postgres adapter over a SQLAlchemy ``Session`` (one session per request).

    ``save_or_get`` commits: one call is one transaction, so a batch gets per-item idempotency.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_or_get(self, record: DecisionRecord) -> SaveResult:
        statement = (
            insert(DecisionRow)
            .values(
                transaction_id=record.transaction_id,
                request_hash=record.request_hash,
                request=record.request,
                score=record.score,
                decision=record.decision,
                model_version=record.model_version,
                reason_codes=record.reason_codes,
                latency_ms=record.latency_ms,
            )
            .on_conflict_do_nothing(index_elements=[DecisionRow.transaction_id])
            .returning(DecisionRow)
        )
        try:
            inserted = self._session.execute(statement).scalar_one_or_none()
            if inserted is not None:
                result = SaveResult(_to_record(inserted), created=True)
            else:
                # Someone else committed this id first (a concurrent call blocks on the unique
                # index until that transaction ends, so by now the row is visible).
                existing = self._session.get(DecisionRow, record.transaction_id)
                assert existing is not None, "ON CONFLICT hit but the row is not visible"
                self._session.refresh(existing)
                result = SaveResult(_to_record(existing), created=False)
            self._session.commit()
        except BaseException:
            self._session.rollback()
            raise
        return result

    def get(self, transaction_id: str) -> DecisionRecord | None:
        row = self._session.execute(
            select(DecisionRow).where(DecisionRow.transaction_id == transaction_id)
        ).scalar_one_or_none()
        self._session.rollback()  # end the read-only transaction
        return _to_record(row) if row is not None else None


def _to_record(row: DecisionRow) -> DecisionRecord:
    return DecisionRecord(
        transaction_id=row.transaction_id,
        request_hash=row.request_hash,
        request=row.request,
        score=row.score,
        decision=row.decision,
        model_version=row.model_version,
        reason_codes=list(row.reason_codes),
        latency_ms=row.latency_ms,
        created_at=row.created_at,
    )
