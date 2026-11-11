import io
import threading
import time
from datetime import date, timedelta

import duckdb
import polars as pl
import pytest

from payments_shared.data_contracts import ContractViolationError, load_contract
from settlement.storage import marts as marts_module
from settlement.storage.marts import MART_NAMES, PART_FILE, STAGING_DIR, MartStore, empty_mart

D1, D2, D3 = date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)


def fee_rows(day: date, n: int = 3, gross: int = 1_000) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "settlement_date": [day] * n,
            "merchant_id": [f"mer_{i:06x}" for i in range(n)][::-1],  # unsorted on purpose
            "currency": ["BRL"] * n,
            "gross_minor": [gross + i for i in range(n)],
            "expected_fee_minor": [10] * n,
            "actual_fee_minor": [11] * n,
            "fee_diff_minor": [1] * n,
        }
    )


@pytest.fixture
def store(tmp_path) -> MartStore:
    return MartStore(tmp_path / "marts")


def test_partition_layout_is_hive_style(store):
    path = store.write_partition("fee_summary", D1, fee_rows(D1))

    assert path == store.root / "fee_summary" / "settlement_date=2026-01-01" / PART_FILE
    assert store.has_partition("fee_summary", D1) and not store.has_partition("fee_summary", D2)


def test_rows_are_stored_sorted_by_primary_key_and_typed_by_the_contract(store):
    store.write_partition("fee_summary", D1, fee_rows(D1))
    frame = store.read_partition("fee_summary", D1)

    assert frame["merchant_id"].to_list() == sorted(frame["merchant_id"].to_list())
    assert frame.schema == load_contract("fee_summary").schema


def test_partitions_are_zstd_compressed(store):
    path = store.write_partition("fee_summary", D1, fee_rows(D1))
    codecs = duckdb.sql(f"select distinct compression from parquet_metadata('{path}')").fetchall()
    assert codecs == [("ZSTD",)]


def test_rerun_is_idempotent(store):
    first = store.write_partition("fee_summary", D1, fee_rows(D1))
    before = first.read_bytes()
    store.write_partition("fee_summary", D1, fee_rows(D1))

    assert first.read_bytes() == before
    assert store.read_partition("fee_summary", D1).equals(fee_rows(D1).sort("merchant_id"))


def test_rewrite_replaces_the_partition(store):
    store.write_partition("fee_summary", D1, fee_rows(D1, n=3))
    store.write_partition("fee_summary", D1, fee_rows(D1, n=1, gross=9_999))

    frame = store.read_partition("fee_summary", D1)
    assert frame.height == 1 and frame["gross_minor"][0] == 9_999


@pytest.mark.parametrize("mart", MART_NAMES)
def test_empty_partitions_have_the_contract_schema(store, mart):
    path = store.write_empty_partition(mart, D1)

    frame = pl.read_parquet(path)
    assert frame.height == 0
    assert frame.schema == load_contract(mart).schema
    assert store.has_partition(mart, D1)  # "ran and found nothing" is visible


def test_a_contract_violation_is_rejected_and_the_old_partition_survives(store):
    store.write_partition("fee_summary", D1, fee_rows(D1))
    before = store.partition_file("fee_summary", D1).read_bytes()
    broken = fee_rows(D1).with_columns(fee_diff_minor=pl.lit(999))  # violates diff_is_consistent

    with pytest.raises(ContractViolationError, match="diff_is_consistent"):
        store.write_partition("fee_summary", D1, broken)

    assert store.partition_file("fee_summary", D1).read_bytes() == before
    assert not (store.root / STAGING_DIR).exists()


def test_a_failure_in_the_middle_of_the_write_leaves_no_trace(store, monkeypatch):
    store.write_partition("fee_summary", D1, fee_rows(D1))
    before = store.partition_file("fee_summary", D1).read_bytes()

    def half_written(self, file, *args, **kwargs):
        file.write_bytes(b"PAR1 truncated")
        raise OSError("disk full")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", half_written)
    with pytest.raises(OSError, match="disk full"):
        store.write_partition("fee_summary", D1, fee_rows(D1, n=1, gross=5))

    assert store.partition_file("fee_summary", D1).read_bytes() == before
    assert not (store.root / STAGING_DIR).exists(), "the staging directory must be cleaned up"


def test_a_failure_while_swapping_keeps_the_previous_partition(store, monkeypatch):
    store.write_partition("fee_summary", D1, fee_rows(D1))
    before = store.partition_file("fee_summary", D1).read_bytes()

    def failing_replace(src, dst):
        raise OSError("rename failed")

    monkeypatch.setattr(marts_module.os, "replace", failing_replace)
    with pytest.raises(OSError, match="rename failed"):
        store.write_partition("fee_summary", D1, fee_rows(D1, n=1))

    assert store.partition_file("fee_summary", D1).read_bytes() == before
    assert not (store.root / STAGING_DIR).exists()


def test_a_failed_first_write_creates_no_partition(store):
    broken = fee_rows(D1).with_columns(gross_minor=pl.lit(-5))
    with pytest.raises(ContractViolationError):
        store.write_partition("fee_summary", D1, broken)
    assert not store.has_partition("fee_summary", D1)
    assert store.partition_dates("fee_summary") == []


def test_rows_of_another_date_are_rejected(store):
    with pytest.raises(ValueError, match="do not belong"):
        store.write_partition("fee_summary", D1, fee_rows(D2))


def test_missing_columns_are_rejected(store):
    with pytest.raises(ValueError, match="missing columns"):
        store.write_partition("fee_summary", D1, fee_rows(D1).drop("gross_minor"))


def test_unknown_marts_are_rejected(store):
    with pytest.raises(ValueError, match="unknown mart"):
        store.partition_dir("../etc", D1)


def test_a_reader_never_sees_a_half_written_partition(store):
    """Overwrite a partition in a loop while another thread reads it: every read is complete."""
    store.write_partition("fee_summary", D1, fee_rows(D1, n=200, gross=1))
    stop = threading.Event()
    seen: set[int] = set()
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            while not stop.is_set():
                # One open() per read: this is what DuckDB or a plain HTTP download does.
                frame = pl.read_parquet(
                    io.BytesIO(store.partition_file("fee_summary", D1).read_bytes())
                )
                seen.add(frame.height)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for i in range(60):
            store.write_partition("fee_summary", D1, fee_rows(D1, n=200 if i % 2 else 50, gross=i))
    finally:
        stop.set()
        thread.join(timeout=10)

    assert not errors, errors
    assert seen <= {50, 200}


# --- reading ----------------------------------------------------------------------------------


@pytest.fixture
def filled(store) -> MartStore:
    for day in (D1, D2, D3):
        store.write_partition("fee_summary", day, fee_rows(day))
    store.write_empty_partition("daily_settlement_by_merchant", D1)
    return store


def test_partition_dates_are_discovered(filled):
    assert filled.partition_dates("fee_summary") == [D1, D2, D3]
    assert filled.partition_dates("reconciliation_breaks") == []


def test_scan_pushes_the_date_range_down_to_the_partitions(filled):
    lazy = filled.scan("fee_summary", from_date=D2, to_date=D2)

    plan = lazy.explain()
    assert "settlement_date=2026-01-02" in plan
    assert "settlement_date=2026-01-01" not in plan and "settlement_date=2026-01-03" not in plan
    assert lazy.collect()["settlement_date"].unique().to_list() == [D2]


def test_scan_of_an_open_range_reads_everything(filled):
    assert filled.scan("fee_summary").collect().height == 9
    assert filled.scan("fee_summary", from_date=D2).collect().height == 6


def test_scan_of_a_mart_without_partitions_is_an_empty_frame_with_the_schema(filled):
    frame = filled.scan("reconciliation_breaks").collect()
    assert frame.height == 0 and frame.schema == load_contract("reconciliation_breaks").schema


def test_empty_and_full_partitions_scan_together(store):
    store.write_partition("fee_summary", D1, fee_rows(D1))
    store.write_empty_partition("fee_summary", D2)
    assert store.scan("fee_summary").collect().height == 3


def test_duckdb_discovers_the_hive_partitions(filled):
    """A BI tool can query the marts directly, with no code of ours in between."""
    glob = f"{filled.mart_dir('fee_summary')}/**/*.parquet"
    rows = duckdb.sql(
        f"select settlement_date, count(*), sum(gross_minor) "
        f"from read_parquet('{glob}', hive_partitioning=true) "
        f"where settlement_date >= date '2026-01-02' group by 1 order by 1"
    ).fetchall()

    assert rows == [(D2, 3, 3_003), (D3, 3, 3_003)]


def test_duckdb_reads_empty_partitions_next_to_full_ones(store):
    store.write_partition("fee_summary", D1, fee_rows(D1))
    store.write_empty_partition("fee_summary", D2)
    glob = f"{store.mart_dir('fee_summary')}/**/*.parquet"
    count = duckdb.sql(f"select count(*) from read_parquet('{glob}', hive_partitioning=true)")
    assert count.fetchall() == [(3,)]


def test_is_readable(tmp_path, store):
    assert not store.is_readable()
    store.root.mkdir()
    assert store.is_readable()
    assert empty_mart("fee_summary").height == 0
    assert timedelta(0) == D1 - D1


def test_polars_readers_survive_a_concurrent_overwrite_by_retrying(store):
    """Polars opens the file twice, so a bare read can hit a swap; the retry helper absorbs it."""
    store.write_partition("fee_summary", D1, fee_rows(D1, n=200, gross=1))
    stop = threading.Event()

    def writer() -> None:
        i = 0
        while not stop.is_set():
            store.write_partition("fee_summary", D1, fee_rows(D1, n=200 if i % 2 else 50, gross=i))
            i += 1
            time.sleep(0.01)  # a busy rewrite loop, but not a pathological one

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        for _ in range(150):
            frame = marts_module.collect_with_retry(lambda: store.scan("fee_summary"))
            assert frame.height in {50, 200}
    finally:
        stop.set()
        thread.join(timeout=10)


def test_retry_gives_up_and_raises_after_the_last_attempt():
    calls = []

    def always_torn():
        calls.append(1)
        raise pl.exceptions.ComputeError("torn")

    with pytest.raises(pl.exceptions.ComputeError):
        marts_module.collect_with_retry(always_torn, attempts=3)
    assert len(calls) == 3
