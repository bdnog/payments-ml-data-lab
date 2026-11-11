"""The contract of ``RunRepository``, run against every adapter (memory and Postgres)."""

import re
import threading
from datetime import UTC, date, datetime, timedelta

import pytest

from settlement.domain import RunStatus
from settlement.domain.runs import MAX_ERROR_LENGTH, IllegalTransitionError
from settlement.persistence.repository import (
    ClaimLostError,
    InMemoryRunRepository,
    RunNotFoundError,
    RunRepository,
)

T0 = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
D1, D2, D3 = date(2026, 1, 15), date(2026, 1, 16), date(2026, 1, 17)


def minutes(n: int) -> datetime:
    return T0 + timedelta(minutes=n)


# --- create: idempotent by date ---------------------------------------------------------------


def test_a_new_run_is_pending(repository: RunRepository):
    result = repository.create(D1, now=T0)

    assert result.created
    run = result.run
    assert re.fullmatch(r"run_[0-9a-f]{12}", run.run_id)
    assert (run.run_date, run.status, run.requested_at) == (D1, RunStatus.PENDING, T0)
    assert (run.started_at, run.finished_at, run.error, run.attempts) == (None, None, None, 0)
    assert repository.get(run.run_id) == run


def test_a_second_request_for_the_date_returns_the_same_run(repository: RunRepository):
    first = repository.create(D1, now=T0)
    second = repository.create(D1, now=minutes(5))

    assert first.created and not second.created
    assert second.run == first.run


def test_a_running_or_done_run_also_blocks_a_new_one(repository: RunRepository):
    first = repository.create(D1, now=T0).run
    claimed = repository.claim_next(now=minutes(1))
    assert claimed and claimed.run_id == first.run_id
    assert not repository.create(D1).created  # RUNNING

    repository.complete(first.run_id, 1, {"rows": 1}, now=minutes(2))
    again = repository.create(D1)
    assert not again.created and again.run.status == RunStatus.DONE


def test_a_failed_run_allows_a_new_one(repository: RunRepository):
    failed = repository.create(D1, now=T0).run
    repository.claim_next(now=minutes(1))
    repository.fail(failed.run_id, 1, "boom", now=minutes(2))

    retry = repository.create(D1, now=minutes(3))

    assert retry.created and retry.run.run_id != failed.run_id
    assert repository.get(failed.run_id).status == RunStatus.FAILED  # history is kept
    assert not repository.create(D1).created  # and the new one blocks again


def test_different_dates_are_independent(repository: RunRepository):
    assert repository.create(D1).created and repository.create(D2).created


def test_get_returns_none_for_an_unknown_run(repository: RunRepository):
    assert repository.get("run_ffffffffffff") is None


# --- claim: oldest pending first --------------------------------------------------------------


def test_claim_takes_the_oldest_pending_run_and_marks_it_running(repository: RunRepository):
    later = repository.create(D2, now=minutes(5)).run
    oldest = repository.create(D1, now=T0).run

    claimed = repository.claim_next(now=minutes(10))

    assert claimed is not None and claimed.run_id == oldest.run_id
    assert claimed.status == RunStatus.RUNNING and claimed.attempts == 1
    assert claimed.started_at == claimed.claimed_at == minutes(10)
    assert repository.get(later.run_id).status == RunStatus.PENDING


def test_claim_returns_none_when_nothing_is_pending(repository: RunRepository):
    assert repository.claim_next() is None
    repository.create(D1)
    repository.claim_next()
    assert repository.claim_next() is None  # the only run is RUNNING now


# --- complete and fail: the state machine -----------------------------------------------------


def test_complete_records_done_with_the_summary(repository: RunRepository):
    run = repository.create(D1, now=T0).run
    repository.claim_next(now=minutes(1))

    done = repository.complete(run.run_id, 1, {"breaks": 3}, now=minutes(2))

    assert done.status == RunStatus.DONE and done.finished_at == minutes(2)
    assert done.summary == {"breaks": 3} and done.error is None
    assert repository.get(run.run_id) == done


def test_fail_records_the_error_truncated_to_2000_characters(repository: RunRepository):
    run = repository.create(D1).run
    repository.claim_next()

    failed = repository.fail(run.run_id, 1, "x" * 5_000)

    assert failed.status == RunStatus.FAILED and failed.finished_at is not None
    assert failed.error is not None and len(failed.error) == MAX_ERROR_LENGTH


def test_a_pending_run_cannot_be_completed_or_failed(repository: RunRepository):
    run = repository.create(D1).run
    with pytest.raises(IllegalTransitionError):
        repository.complete(run.run_id, 0, {})
    with pytest.raises(IllegalTransitionError):
        repository.fail(run.run_id, 0, "no")
    assert repository.get(run.run_id).status == RunStatus.PENDING


def test_finished_runs_are_terminal(repository: RunRepository):
    run = repository.create(D1).run
    repository.claim_next()
    repository.complete(run.run_id, 1, {})

    with pytest.raises(IllegalTransitionError, match="DONE to DONE"):
        repository.complete(run.run_id, 1, {})
    with pytest.raises(IllegalTransitionError, match="DONE to FAILED"):
        repository.fail(run.run_id, 1, "late")
    assert repository.get(run.run_id).status == RunStatus.DONE


def test_an_unknown_run_cannot_be_finished(repository: RunRepository):
    with pytest.raises(RunNotFoundError):
        repository.complete("run_ffffffffffff", 1, {})


def test_a_worker_that_lost_its_run_cannot_finish_it(repository: RunRepository):
    """Attempt 1 went stale and attempt 2 now owns the run: attempt 1 must not overwrite it."""
    run = repository.create(D1, now=T0).run
    repository.claim_next(now=minutes(1))
    repository.recover_stale(stale_before=minutes(30), max_attempts=3, now=minutes(31))
    second = repository.claim_next(now=minutes(32))
    assert second is not None and second.attempts == 2

    with pytest.raises(ClaimLostError):
        repository.complete(run.run_id, 1, {"from": "the slow worker"})

    assert repository.get(run.run_id).status == RunStatus.RUNNING
    assert repository.complete(run.run_id, 2, {"from": "the owner"}).summary == {
        "from": "the owner"
    }


# --- stale recovery ---------------------------------------------------------------------------


def test_a_stale_running_run_goes_back_to_pending(repository: RunRepository):
    run = repository.create(D1, now=T0).run
    repository.claim_next(now=minutes(1))

    recovered = repository.recover_stale(stale_before=minutes(16), max_attempts=3, now=minutes(20))

    assert [r.run_id for r in recovered] == [run.run_id]
    back = repository.get(run.run_id)
    assert back.status == RunStatus.PENDING and back.attempts == 1
    assert back.started_at is None and back.claimed_at is None, "a PENDING run has not started"
    reclaimed = repository.claim_next(now=minutes(21))
    assert reclaimed is not None and reclaimed.attempts == 2


def test_a_run_that_is_still_fresh_is_left_alone(repository: RunRepository):
    run = repository.create(D1, now=T0).run
    repository.claim_next(now=minutes(10))

    assert repository.recover_stale(stale_before=minutes(5), max_attempts=3) == []
    assert repository.get(run.run_id).status == RunStatus.RUNNING


def test_a_stale_run_out_of_attempts_fails(repository: RunRepository):
    run = repository.create(D1, now=T0).run
    for attempt in range(1, 4):
        claimed = repository.claim_next(now=minutes(attempt * 100))
        assert claimed is not None and claimed.attempts == attempt
        repository.recover_stale(
            stale_before=minutes(attempt * 100 + 50),
            max_attempts=3,
            now=minutes(attempt * 100 + 60),
        )

    final = repository.get(run.run_id)
    assert final.status == RunStatus.FAILED and final.attempts == 3
    assert final.error == "abandoned by its worker and out of attempts (3 of 3)"
    assert final.finished_at is not None
    assert repository.claim_next() is None


def test_recovery_only_touches_stale_running_runs(repository: RunRepository):
    pending = repository.create(D3, now=minutes(3)).run
    done = repository.create(D1, now=minutes(1)).run
    stuck = repository.create(D2, now=minutes(2)).run
    first = repository.claim_next(now=minutes(10))  # D1, the oldest
    assert first is not None and first.run_id == done.run_id
    repository.complete(done.run_id, 1, {})
    second = repository.claim_next(now=minutes(11))  # D2: claimed and then abandoned
    assert second is not None and second.run_id == stuck.run_id

    recovered = repository.recover_stale(stale_before=minutes(500), max_attempts=3)

    assert [r.run_id for r in recovered] == [stuck.run_id]
    assert repository.get(done.run_id).status == RunStatus.DONE
    assert repository.get(pending.run_id).status == RunStatus.PENDING
    assert repository.get(stuck.run_id).status == RunStatus.PENDING


# --- concurrency ------------------------------------------------------------------------------


def test_two_workers_claiming_concurrently_run_every_run_exactly_once(repository: RunRepository):
    n_runs = 24
    for i in range(n_runs):
        repository.create(date(2026, 1, 1) + timedelta(days=i), now=minutes(i))

    barrier = threading.Barrier(2)
    claimed: list[list[str]] = [[], []]
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            barrier.wait(timeout=10)
            while (run := repository.claim_next()) is not None:
                claimed[index].append(run.run_id)
                repository.complete(run.run_id, run.attempts, {"worker": index})
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not errors, errors
    everything = claimed[0] + claimed[1]
    assert len(everything) == n_runs, "every run was executed"
    assert len(set(everything)) == n_runs, "...and none was executed twice"
    if not isinstance(repository, InMemoryRunRepository):
        assert claimed[0] and claimed[1], "both workers took part (SKIP LOCKED spreads the work)"


def test_concurrent_requests_for_one_date_create_one_run(repository: RunRepository):
    n_threads = 10
    barrier = threading.Barrier(n_threads)
    results = []
    errors: list[BaseException] = []

    def request() -> None:
        try:
            barrier.wait(timeout=10)
            results.append(repository.create(D1))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=request) for _ in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert sum(r.created for r in results) == 1
    assert len({r.run.run_id for r in results}) == 1
