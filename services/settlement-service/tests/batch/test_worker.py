import signal
import threading
from datetime import UTC, date, datetime, timedelta

from settlement.batch.worker import (
    Worker,
    install_signal_handlers,
    pipeline_executor,
)
from settlement.config import Settings
from settlement.domain import RunStatus
from settlement.persistence.repository import InMemoryRunRepository

T0 = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now


def queue(*days: int) -> InMemoryRunRepository:
    repository = InMemoryRunRepository()
    for i, day in enumerate(days):
        repository.create(date(2026, 1, day), now=T0 + timedelta(seconds=i))
    return repository


def statuses(repository: InMemoryRunRepository) -> list[RunStatus]:
    return [r.status for r in sorted(repository._runs.values(), key=lambda r: r.run_date)]


def test_once_drains_the_queue_and_records_the_summaries():
    repository = queue(1, 2, 3)
    seen: list[date] = []
    worker = Worker(
        repository, lambda d: (seen.append(d), {"date": d.isoformat()})[1], clock=Clock()
    )

    assert worker.drain() == 3

    assert seen == [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)], "oldest request first"
    assert statuses(repository) == [RunStatus.DONE] * 3
    assert {r.summary["date"] for r in repository._runs.values()} == {
        "2026-01-01", "2026-01-02", "2026-01-03",
    }  # fmt: skip
    assert worker.run_once() is False


def test_a_failing_pipeline_marks_the_run_failed_with_the_error_and_the_worker_goes_on():
    repository = queue(1, 2)

    def execute(day: date) -> dict:
        if day.day == 1:
            raise ValueError("the PSP file is corrupt")
        return {"ok": True}

    Worker(repository, execute, clock=Clock()).drain()

    first, second = sorted(repository._runs.values(), key=lambda r: r.run_date)
    assert first.status == RunStatus.FAILED and first.error == "ValueError: the PSP file is corrupt"
    assert first.finished_at is not None and first.summary is None
    assert second.status == RunStatus.DONE


def test_a_huge_error_message_is_truncated():
    repository = queue(1)

    def execute(day: date) -> dict:
        raise RuntimeError("x" * 10_000)

    Worker(repository, execute).drain()

    error = next(iter(repository._runs.values())).error
    assert error is not None and len(error) == 2_000


def test_a_stale_run_is_recovered_and_executed_again():
    clock = Clock()
    repository = queue(1)
    repository.claim_next(now=T0)  # a worker took it... and died
    clock.now = T0 + timedelta(minutes=20)
    executed: list[date] = []

    worker = Worker(
        repository, lambda d: (executed.append(d), {})[1], stale_after_seconds=900, clock=clock
    )
    worker.drain()

    run = next(iter(repository._runs.values()))
    assert executed == [date(2026, 1, 1)]
    assert run.status == RunStatus.DONE and run.attempts == 2, "at-least-once: it ran again"


def test_a_run_that_is_not_stale_yet_is_left_to_its_worker():
    clock = Clock()
    repository = queue(1)
    repository.claim_next(now=T0)
    clock.now = T0 + timedelta(minutes=5)

    assert (
        Worker(repository, lambda d: {}, stale_after_seconds=900, clock=clock).run_once() is False
    )
    assert statuses(repository) == [RunStatus.RUNNING]


def test_a_run_out_of_attempts_is_failed_instead_of_retried_forever():
    clock = Clock()
    repository = queue(1)
    for attempt in range(1, 4):  # three workers took it and died
        repository.claim_next(now=T0 + timedelta(hours=attempt))
        repository.recover_stale(T0 + timedelta(hours=attempt, minutes=30), 99)
    repository.claim_next(now=T0 + timedelta(hours=4))
    clock.now = T0 + timedelta(hours=6)
    executed: list = []

    Worker(repository, lambda d: executed.append(d) or {}, max_attempts=3, clock=clock).drain()

    run = next(iter(repository._runs.values()))
    assert executed == [] and run.status == RunStatus.FAILED
    assert "out of attempts" in run.error


def test_sigterm_finishes_the_current_run_and_claims_no_other():
    repository = queue(1, 2, 3)
    stop = threading.Event()

    def execute(day: date) -> dict:
        if day.day == 1:
            stop.set()  # what the SIGTERM handler does, in the middle of the first run
        return {"done": day.day}

    worker = Worker(repository, execute, stop=stop)
    processed = worker.run_forever()

    assert processed == 1
    assert statuses(repository) == [RunStatus.DONE, RunStatus.PENDING, RunStatus.PENDING]


def test_an_idle_worker_sleeps_for_the_poll_interval_and_stops_when_asked():
    waits: list[float] = []

    class Stop(threading.Event):
        def wait(self, timeout=None) -> bool:
            waits.append(timeout)
            if len(waits) == 3:
                self.set()
            return super().wait(0)

    worker = Worker(InMemoryRunRepository(), lambda d: {}, poll_seconds=7.5, stop=Stop())

    assert worker.run_forever() == 0
    assert waits == [7.5, 7.5, 7.5]


def test_a_worker_that_lost_its_run_does_not_crash_or_overwrite_the_newer_attempt():
    clock = Clock()
    repository = queue(1)

    def slow(day: date) -> dict:
        # While we compute, another worker takes the run over (we were declared stale).
        clock.now = T0 + timedelta(hours=1)
        repository.recover_stale(clock.now - timedelta(minutes=15), 5, clock.now)
        repository.claim_next(clock.now)
        return {"from": "the slow worker"}

    Worker(repository, slow, clock=clock).run_once()

    run = next(iter(repository._runs.values()))
    assert run.status == RunStatus.RUNNING and run.attempts == 2 and run.summary is None


def test_sigterm_and_sigint_set_the_stop_flag():
    stop = threading.Event()
    restore = install_signal_handlers(stop)
    try:
        signal.raise_signal(signal.SIGTERM)
        assert stop.is_set()
        stop.clear()
        signal.raise_signal(signal.SIGINT)
        assert stop.is_set()
    finally:
        restore()


# --- the real pipeline behind the worker ------------------------------------------------------


def test_the_worker_runs_the_real_pipeline_for_the_requested_date(sample_dir, tmp_path):
    settings = Settings(input_dir=sample_dir, marts_dir=tmp_path / "marts")
    repository = InMemoryRunRepository()
    run = repository.create(date(2026, 1, 15)).run

    Worker(repository, pipeline_executor(settings)).drain()

    done = repository.get(run.run_id)
    assert done.status == RunStatus.DONE
    assert done.summary["first_date"] == "2026-01-15" and done.summary["date_count"] == 1
    assert set(done.summary["breaks_by_type"]) >= {"AMOUNT_MISMATCH", "MISSING_IN_PSP"}
    assert (tmp_path / "marts" / "fee_summary" / "settlement_date=2026-01-15").is_dir()


def test_a_missing_input_dataset_fails_the_run_with_a_helpful_message(tmp_path):
    settings = Settings(input_dir=tmp_path / "nothing", marts_dir=tmp_path / "marts")
    repository = InMemoryRunRepository()
    repository.create(date(2026, 1, 15))

    Worker(repository, pipeline_executor(settings)).drain()

    failed = next(iter(repository._runs.values()))
    assert failed.status == RunStatus.FAILED
    assert "FileNotFoundError" in failed.error and "payments_datagen generate" in failed.error
