"""The run-queue port and its two adapters (in memory, Postgres).

``RunRepository`` is the *port*; ``InMemoryRunRepository`` and ``SqlRunRepository`` are the
*adapters*, and one contract-test suite runs against both. The operations are shaped for a queue:

* ``create`` is insert-first and idempotent by date (a partial unique index decides);
* ``claim_next`` takes the oldest PENDING run with ``FOR UPDATE SKIP LOCKED``, so any number of
  workers can poll the same table and each run goes to exactly one of them;
* ``complete`` and ``fail`` are guarded by status *and* attempt number, so a slow worker whose run
  was already given to another one cannot overwrite the newer attempt;
* ``recover_stale`` gives abandoned RUNNING runs back to the queue, or fails them when out of tries.

Every operation is its own transaction. Delivery is *at least once* (a crashed worker's run runs
again), which is safe because the pipeline is idempotent (ADR 0012).
"""

import secrets
import threading
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from typing import Any, Protocol

from sqlalchemy import func, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from settlement.domain import RunStatus
from settlement.domain.runs import IllegalTransitionError, check_transition, truncate_error
from settlement.persistence.models import RunRow

ABANDONED_MESSAGE = "abandoned by its worker and out of attempts ({attempts} of {max_attempts})"


class RunNotFoundError(Exception):
    """No run has this id."""


class ClaimLostError(Exception):
    """The run is RUNNING but belongs to a later attempt: this worker no longer owns it."""


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    run_date: date
    status: RunStatus
    requested_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    attempts: int = 0
    claimed_at: datetime | None = None
    summary: dict[str, Any] | None = None


@dataclass(frozen=True)
class CreateResult:
    run: RunRecord
    created: bool  # False: the date already had a run that has not failed


class RunRepository(Protocol):
    def create(self, run_date: date, now: datetime | None = None) -> CreateResult: ...

    def get(self, run_id: str) -> RunRecord | None: ...

    def claim_next(self, now: datetime | None = None) -> RunRecord | None: ...

    def complete(
        self, run_id: str, attempt: int, summary: dict[str, Any], now: datetime | None = None
    ) -> RunRecord: ...

    def fail(
        self, run_id: str, attempt: int, error: str, now: datetime | None = None
    ) -> RunRecord: ...

    def recover_stale(
        self, stale_before: datetime, max_attempts: int, now: datetime | None = None
    ) -> list[RunRecord]: ...


def new_run_id() -> str:
    return f"run_{secrets.token_hex(6)}"


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(UTC)


class InMemoryRunRepository:
    """A dict behind a lock, with the same semantics as the SQL adapter."""

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.Lock()

    def create(self, run_date: date, now: datetime | None = None) -> CreateResult:
        with self._lock:
            for run in self._runs.values():
                if run.run_date == run_date and run.status != RunStatus.FAILED:
                    return CreateResult(run, created=False)
            run = RunRecord(new_run_id(), run_date, RunStatus.PENDING, _now(now))
            self._runs[run.run_id] = run
            return CreateResult(run, created=True)

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def claim_next(self, now: datetime | None = None) -> RunRecord | None:
        with self._lock:
            pending = [r for r in self._runs.values() if r.status == RunStatus.PENDING]
            if not pending:
                return None
            oldest = min(pending, key=lambda r: (r.requested_at, r.run_id))
            moment = _now(now)
            claimed = replace(
                oldest,
                status=RunStatus.RUNNING,
                attempts=oldest.attempts + 1,
                started_at=moment,
                claimed_at=moment,
            )
            self._runs[claimed.run_id] = claimed
            return claimed

    def _finish(
        self, run_id: str, attempt: int, target: RunStatus, now: datetime | None, **changes: Any
    ) -> RunRecord:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise RunNotFoundError(run_id)
            check_transition(run.status, target)
            if run.attempts != attempt:
                raise ClaimLostError(f"{run_id} is on attempt {run.attempts}, not {attempt}")
            finished = replace(run, status=target, finished_at=_now(now), **changes)
            self._runs[run_id] = finished
            return finished

    def complete(
        self, run_id: str, attempt: int, summary: dict[str, Any], now: datetime | None = None
    ) -> RunRecord:
        return self._finish(run_id, attempt, RunStatus.DONE, now, summary=summary)

    def fail(self, run_id: str, attempt: int, error: str, now: datetime | None = None) -> RunRecord:
        return self._finish(run_id, attempt, RunStatus.FAILED, now, error=truncate_error(error))

    def recover_stale(
        self, stale_before: datetime, max_attempts: int, now: datetime | None = None
    ) -> list[RunRecord]:
        recovered: list[RunRecord] = []
        with self._lock:
            for run in list(self._runs.values()):
                if (
                    run.status != RunStatus.RUNNING
                    or run.claimed_at is None
                    or run.claimed_at >= stale_before
                ):
                    continue
                if run.attempts >= max_attempts:
                    message = ABANDONED_MESSAGE.format(
                        attempts=run.attempts, max_attempts=max_attempts
                    )
                    updated = replace(
                        run, status=RunStatus.FAILED, finished_at=_now(now), error=message
                    )
                else:
                    updated = replace(
                        run, status=RunStatus.PENDING, started_at=None, claimed_at=None
                    )
                self._runs[run.run_id] = updated
                recovered.append(updated)
        return recovered


class SqlRunRepository:
    """Postgres adapter: each method opens its own session and commits, one call per transaction."""

    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def create(self, run_date: date, now: datetime | None = None) -> CreateResult:
        for _ in range(3):  # a conflicting run can turn FAILED between the insert and the read
            with self._sessions.begin() as session:
                inserted = session.execute(
                    insert(RunRow)
                    .values(
                        run_id=new_run_id(),
                        run_date=run_date,
                        status=RunStatus.PENDING.value,
                        requested_at=_now(now),
                        attempts=0,
                    )
                    .on_conflict_do_nothing(
                        index_elements=[RunRow.run_date], index_where=text("status <> 'FAILED'")
                    )
                    .returning(RunRow)
                ).scalar_one_or_none()
                if inserted is not None:
                    return CreateResult(_to_record(inserted), created=True)
                existing = session.execute(
                    select(RunRow).where(RunRow.run_date == run_date, RunRow.status != "FAILED")
                ).scalar_one_or_none()
                if existing is not None:
                    return CreateResult(_to_record(existing), created=False)
        raise RuntimeError(f"could not create or find a run for {run_date}")

    def get(self, run_id: str) -> RunRecord | None:
        with self._sessions() as session:
            row = session.get(RunRow, run_id)
            return _to_record(row) if row is not None else None

    def claim_next(self, now: datetime | None = None) -> RunRecord | None:
        moment = _now(now)
        oldest_pending = (
            select(RunRow.run_id)
            .where(RunRow.status == RunStatus.PENDING.value)
            .order_by(RunRow.requested_at, RunRow.run_id)
            .limit(1)
            .with_for_update(
                skip_locked=True
            )  # rows another worker holds are skipped, not waited on
            .scalar_subquery()
        )
        with self._sessions.begin() as session:
            row = session.execute(
                update(RunRow)
                .where(RunRow.run_id == oldest_pending)
                .values(
                    status=RunStatus.RUNNING.value,
                    attempts=RunRow.attempts + 1,
                    started_at=moment,
                    claimed_at=moment,
                )
                .returning(RunRow)
            ).scalar_one_or_none()
            return _to_record(row) if row is not None else None

    def _finish(
        self, run_id: str, attempt: int, target: RunStatus, now: datetime | None, **values: Any
    ) -> RunRecord:
        with self._sessions.begin() as session:
            # The WHERE clause is the state machine: only a RUNNING run of this very attempt moves.
            row = session.execute(
                update(RunRow)
                .where(
                    RunRow.run_id == run_id,
                    RunRow.status == RunStatus.RUNNING.value,
                    RunRow.attempts == attempt,
                )
                .values(status=target.value, finished_at=_now(now), **values)
                .returning(RunRow)
            ).scalar_one_or_none()
            if row is not None:
                return _to_record(row)
            current = session.get(RunRow, run_id)
        if current is None:
            raise RunNotFoundError(run_id)
        status = RunStatus(current.status)
        if status != RunStatus.RUNNING:
            raise IllegalTransitionError(status, target)
        raise ClaimLostError(f"{run_id} is on attempt {current.attempts}, not {attempt}")

    def complete(
        self, run_id: str, attempt: int, summary: dict[str, Any], now: datetime | None = None
    ) -> RunRecord:
        return self._finish(run_id, attempt, RunStatus.DONE, now, summary=summary)

    def fail(self, run_id: str, attempt: int, error: str, now: datetime | None = None) -> RunRecord:
        return self._finish(run_id, attempt, RunStatus.FAILED, now, error=truncate_error(error))

    def recover_stale(
        self, stale_before: datetime, max_attempts: int, now: datetime | None = None
    ) -> list[RunRecord]:
        stale = (RunRow.status == RunStatus.RUNNING.value) & (RunRow.claimed_at < stale_before)
        with self._sessions.begin() as session:
            out_of_attempts = (
                session.execute(
                    update(RunRow)
                    .where(stale, RunRow.attempts >= max_attempts)
                    .values(
                        status=RunStatus.FAILED.value,
                        finished_at=_now(now),
                        error=func.concat(
                            "abandoned by its worker and out of attempts (",
                            RunRow.attempts,
                            f" of {int(max_attempts)})",
                        ),
                    )
                    .returning(RunRow)
                )
                .scalars()
                .all()
            )
            retried = (
                session.execute(
                    update(RunRow)
                    .where(stale, RunRow.attempts < max_attempts)
                    .values(status=RunStatus.PENDING.value, started_at=None, claimed_at=None)
                    .returning(RunRow)
                )
                .scalars()
                .all()
            )
            return [_to_record(r) for r in (*out_of_attempts, *retried)]


def _to_record(row: RunRow) -> RunRecord:
    return RunRecord(
        run_id=row.run_id,
        run_date=row.run_date,
        status=RunStatus(row.status),
        requested_at=row.requested_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
        error=row.error,
        attempts=row.attempts,
        claimed_at=row.claimed_at,
        summary=row.summary,
    )
