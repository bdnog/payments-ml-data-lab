"""The worker against a real Postgres queue, through the same CLI entry point the image uses."""

from datetime import date

import pytest

from settlement.batch.__main__ import main
from settlement.domain import RunStatus
from settlement.persistence.repository import SqlRunRepository
from settlement.persistence.session import session_factory

from .conftest import SKIP_SQL, TEST_DATABASE_URL

pytestmark = SKIP_SQL


@pytest.fixture
def configured(monkeypatch, clean_table, sample_dir, tmp_path):
    monkeypatch.setenv("SETTLEMENT_DATABASE_URL", TEST_DATABASE_URL)
    monkeypatch.setenv("SETTLEMENT_INPUT_DIR", str(sample_dir))
    monkeypatch.setenv("SETTLEMENT_MARTS_DIR", str(tmp_path / "marts"))
    return SqlRunRepository(session_factory(clean_table)), tmp_path / "marts"


def test_worker_once_executes_queued_runs_and_records_the_summary(configured):
    repository, marts = configured
    first = repository.create(date(2026, 1, 14)).run
    second = repository.create(date(2026, 1, 15)).run

    assert main(["worker", "--once"]) == 0

    for run, day in ((first, "2026-01-14"), (second, "2026-01-15")):
        done = repository.get(run.run_id)
        assert done.status == RunStatus.DONE and done.attempts == 1
        assert done.summary["first_date"] == day
        assert (
            marts / "reconciliation_breaks" / f"settlement_date={day}" / "part-0.parquet"
        ).is_file()


def test_worker_once_on_an_empty_queue_exits_cleanly(configured):
    assert main(["worker", "--once"]) == 0


def test_worker_once_records_a_failing_run(configured, monkeypatch, tmp_path):
    repository, _ = configured
    monkeypatch.setenv("SETTLEMENT_INPUT_DIR", str(tmp_path / "no-such-dataset"))
    run = repository.create(date(2026, 1, 14)).run

    assert main(["worker", "--once"]) == 0  # a failed run is a result, not a worker crash

    failed = repository.get(run.run_id)
    assert failed.status == RunStatus.FAILED and "input dataset(s) not found" in failed.error
