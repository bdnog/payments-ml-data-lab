"""The worker: claims queued runs from the ``runs`` table and executes the pipeline.

The API only *registers* a run (``POST /v1/runs``); a worker does the work. Any number of workers
can poll the same table: ``claim_next`` uses ``FOR UPDATE SKIP LOCKED``, so each run goes to one.

* **At-least-once.** If a worker dies mid-run the run stays RUNNING until it is *stale*
  (``stale_after_seconds`` without finishing); the next worker to poll gives it back to the queue
  (or fails it after ``max_attempts``) and it runs again. That is safe because the pipeline is
  idempotent. A run that is merely slower than ``stale_after_seconds`` is also re-queued and may run
  twice; attempt-guarded ``complete`` and ``fail`` keep the older attempt from overwriting the
  newer.
* **Graceful shutdown.** SIGTERM (what ``docker stop`` sends) or SIGINT only sets a flag: the run
  in progress finishes and is recorded, then the worker exits without claiming another.
"""

import logging
import signal
import threading
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from settlement.batch.pipeline import run_dates
from settlement.config import Settings
from settlement.domain.runs import IllegalTransitionError
from settlement.persistence.repository import (
    ClaimLostError,
    RunRecord,
    RunRepository,
    SqlRunRepository,
)
from settlement.persistence.session import create_db_engine, session_factory

log = logging.getLogger("settlement.worker")

Executor = Callable[[date], dict[str, Any]]
BACKOFF_SECONDS = 5.0


class Worker:
    def __init__(
        self,
        repository: RunRepository,
        execute: Executor,
        *,
        poll_seconds: float = 2.0,
        stale_after_seconds: int = 900,
        max_attempts: int = 3,
        stop: threading.Event | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.repository = repository
        self.execute = execute
        self.poll_seconds = poll_seconds
        self.stale_after = timedelta(seconds=stale_after_seconds)
        self.max_attempts = max_attempts
        self.stop = stop or threading.Event()
        self.clock = clock

    # --- one step -------------------------------------------------------------------------------

    def recover_stale(self) -> None:
        now = self.clock()
        for run in self.repository.recover_stale(now - self.stale_after, self.max_attempts, now):
            log.warning(
                "stale run %s (%s) is now %s after %d attempt(s)",
                run.run_id,
                run.run_date,
                run.status,
                run.attempts,
            )

    def run_once(self) -> bool:
        """Recover stale runs, claim at most one run and execute it. False if there was none."""
        self.recover_stale()
        run = self.repository.claim_next(self.clock())
        if run is None:
            return False
        self._execute(run)
        return True

    def _execute(self, run: RunRecord) -> None:
        log.info("run %s for %s started (attempt %d)", run.run_id, run.run_date, run.attempts)
        try:
            summary = self.execute(run.run_date)
        except Exception as exc:  # noqa: BLE001 - whatever went wrong is the run's result
            log.exception("run %s failed", run.run_id)
            self._record_failure(run, f"{type(exc).__name__}: {exc}")
            return
        self._finish(
            lambda: self.repository.complete(run.run_id, run.attempts, summary, self.clock()), run
        )
        log.info("run %s for %s done", run.run_id, run.run_date)

    def _record_failure(self, run: RunRecord, error: str) -> None:
        self._finish(
            lambda: self.repository.fail(run.run_id, run.attempts, error, self.clock()), run
        )

    @staticmethod
    def _finish(record: Callable[[], RunRecord], run: RunRecord) -> None:
        try:
            record()
        except (ClaimLostError, IllegalTransitionError) as exc:
            # Another worker owns this run now (we were too slow): its result stands.
            log.warning("run %s was not recorded: %s", run.run_id, exc)

    # --- loops ----------------------------------------------------------------------------------

    def drain(self) -> int:
        """Execute queued runs until none is left (``worker --once``). Returns how many ran."""
        processed = 0
        while not self.stop.is_set() and self.run_once():
            processed += 1
        return processed

    def run_forever(self) -> int:
        """Poll until asked to stop; sleep ``poll_seconds`` when idle. Returns how many ran."""
        processed = 0
        while not self.stop.is_set():
            try:
                if self.run_once():
                    processed += 1
                    continue
            except SQLAlchemyError:
                log.exception("database unavailable, retrying in %.0fs", BACKOFF_SECONDS)
                self.stop.wait(BACKOFF_SECONDS)
                continue
            self.stop.wait(self.poll_seconds)
        log.info("worker stopped after %d run(s)", processed)
        return processed


def install_signal_handlers(stop: threading.Event) -> Callable[[], None]:
    """Make SIGTERM and SIGINT request a graceful stop. Returns a function that undoes it."""
    previous = {
        sig: signal.signal(sig, lambda signum, _frame: _request_stop(stop, signum))
        for sig in (signal.SIGTERM, signal.SIGINT)
    }

    def restore() -> None:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    return restore


def _request_stop(stop: threading.Event, signum: int) -> None:
    log.info("received %s: finishing the current run, then stopping", signal.Signals(signum).name)
    stop.set()


def pipeline_executor(settings: Settings) -> Executor:
    input_dir, marts_dir = settings.resolved_input_dir(), settings.resolved_marts_dir()

    def execute(run_date: date) -> dict[str, Any]:
        started = time.perf_counter()
        summary = run_dates([run_date], input_dir, marts_dir).to_dict()
        summary["wall_seconds"] = round(time.perf_counter() - started, 3)
        return summary

    return execute


def run_worker(settings: Settings, *, once: bool) -> int:
    engine = create_db_engine(settings.database_url)
    worker = Worker(
        SqlRunRepository(session_factory(engine)),
        pipeline_executor(settings),
        poll_seconds=settings.worker_poll_seconds,
        stale_after_seconds=settings.stale_after_seconds,
        max_attempts=settings.max_attempts,
    )
    restore = install_signal_handlers(worker.stop)
    try:
        processed = worker.drain() if once else worker.run_forever()
        log.info("worker finished: %d run(s) processed", processed)
        return 0
    finally:
        restore()
        engine.dispose()
