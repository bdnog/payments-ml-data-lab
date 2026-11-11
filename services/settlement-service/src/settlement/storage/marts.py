"""Hive-partitioned Parquet marts, written atomically.

Layout (one file per partition, zstd, rows sorted by the mart's primary key)::

    <root>/<mart>/settlement_date=YYYY-MM-DD/part-0.parquet

``settlement_date=...`` in the path is the Hive convention: Polars, DuckDB, Spark and most BI tools
discover it as a column and skip partitions a filter excludes. The column is also kept inside the
file, so an exported single file is self-describing.

**Atomic overwrite.** A partition is rewritten as: write to a staging directory under
``<root>/.staging`` (outside every mart, so readers never glob it), read the file back and validate
it against the mart's data contract, then ``os.replace`` it onto the final path. ``os.replace`` of
a file is atomic on one filesystem, so a reader sees the whole old file or the whole new one, never
a mix and never nothing; a crash or a validation failure leaves the previous partition untouched.
This is why a partition is a single file (see ADR 0011).

**A caveat for readers.** The swap is atomic for any reader that opens the file once (a plain
``open``, DuckDB, ``FileResponse``). Polars, however, opens a local file more than once while it
reads it (footer, then data), so a Polars read that overlaps a swap can fail with a ``ComputeError``
("page header reported the wrong page size"): measured at about one in four reads in a tight loop
against a tight rewrite loop. ``collect_with_retry`` rebuilds and retries the query, which is
correct because the error is transient and the next attempt sees a whole file.

**Empty partitions.** A date with nothing to report is written as an empty file with the right
schema, so "the run happened and found nothing" differs from "the run never happened".
"""

import contextlib
import os
import shutil
import time
import uuid
from collections.abc import Callable
from datetime import date
from pathlib import Path

import polars as pl

from payments_shared.data_contracts import Contract, assert_valid, load_contract

READ_ATTEMPTS = 6
PARTITION_COLUMN = "settlement_date"
PART_FILE = "part-0.parquet"
STAGING_DIR = ".staging"
MART_NAMES = ("daily_settlement_by_merchant", "reconciliation_breaks", "fee_summary")


def conform(frame: pl.DataFrame, contract: Contract) -> pl.DataFrame:
    """The frame with exactly the contract's columns, in order and dtype, sorted by primary key."""
    missing = [c for c in contract.column_names if c not in frame.columns]
    if missing:
        raise ValueError(f"{contract.name}: missing columns {missing}")
    columns = [pl.col(c.name).cast(c.dtype) for c in contract.columns]
    return frame.select(columns).sort(contract.primary_key)


def empty_mart(mart: str) -> pl.DataFrame:
    return pl.DataFrame(schema=load_contract(mart).schema)


class MartStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # --- paths ----------------------------------------------------------------------------------

    def mart_dir(self, mart: str) -> Path:
        self._check_mart(mart)
        return self.root / mart

    def partition_dir(self, mart: str, partition: date) -> Path:
        return self.mart_dir(mart) / f"{PARTITION_COLUMN}={partition.isoformat()}"

    def partition_file(self, mart: str, partition: date) -> Path:
        return self.partition_dir(mart, partition) / PART_FILE

    def has_partition(self, mart: str, partition: date) -> bool:
        return self.partition_file(mart, partition).is_file()

    def partition_dates(self, mart: str) -> list[date]:
        directory = self.mart_dir(mart)
        if not directory.is_dir():
            return []
        prefix = f"{PARTITION_COLUMN}="
        dates = []
        for child in directory.iterdir():
            if child.name.startswith(prefix) and (child / PART_FILE).is_file():
                dates.append(date.fromisoformat(child.name.removeprefix(prefix)))
        return sorted(dates)

    @staticmethod
    def _check_mart(mart: str) -> None:
        if mart not in MART_NAMES:
            raise ValueError(f"unknown mart {mart!r}")

    # --- writing --------------------------------------------------------------------------------

    def write_partition(self, mart: str, partition: date, frame: pl.DataFrame) -> Path:
        """Replace one partition atomically; on any error the old partition is left intact."""
        contract = load_contract(mart)
        frame = conform(frame, contract)
        wrong = frame.filter(
            pl.col(PARTITION_COLUMN).is_not_null() & (pl.col(PARTITION_COLUMN) != partition)
        )
        if wrong.height:
            raise ValueError(
                f"{mart}: {wrong.height} row(s) do not belong to partition {partition.isoformat()}"
            )

        staging = self.root / STAGING_DIR / uuid.uuid4().hex
        staging.mkdir(parents=True)
        try:
            staged = staging / PART_FILE
            frame.write_parquet(staged, compression="zstd")
            assert_valid(pl.read_parquet(staged), contract)  # validate what readers will see
            target = self.partition_file(mart, partition)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, target)
            return target
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            self._remove_staging_root_if_empty()

    def write_empty_partition(self, mart: str, partition: date) -> Path:
        return self.write_partition(mart, partition, empty_mart(mart))

    def _remove_staging_root_if_empty(self) -> None:
        root = self.root / STAGING_DIR
        with contextlib.suppress(OSError):
            root.rmdir()  # only succeeds when empty (no concurrent writer is staging)

    # --- reading --------------------------------------------------------------------------------

    def scan(
        self, mart: str, from_date: date | None = None, to_date: date | None = None
    ) -> pl.LazyFrame:
        """Lazy Hive scan with the date range pushed down: excluded partitions are never opened."""
        directory = self.mart_dir(mart)
        schema = load_contract(mart).schema
        if not self.partition_dates(mart):
            return pl.DataFrame(schema=schema).lazy()
        lazy = pl.scan_parquet(str(directory / "**" / "*.parquet"), hive_partitioning=True)
        if from_date is not None:
            lazy = lazy.filter(pl.col(PARTITION_COLUMN) >= from_date)
        if to_date is not None:
            lazy = lazy.filter(pl.col(PARTITION_COLUMN) <= to_date)
        return lazy

    def read_partition(self, mart: str, partition: date) -> pl.DataFrame:
        return pl.read_parquet(self.partition_file(mart, partition))

    def is_readable(self) -> bool:
        return self.root.is_dir() and os.access(self.root, os.R_OK | os.X_OK)


def collect_with_retry(
    build: Callable[[], pl.LazyFrame], attempts: int = READ_ATTEMPTS
) -> pl.DataFrame:
    """Build and collect a query, retrying when a partition was swapped while Polars was reading.

    Takes a *function that builds* the query, not the query: ``scan_parquet`` already opens the
    files to read their footers, so the race can happen before ``collect()`` is even called.
    """
    for attempt in range(1, attempts + 1):
        try:
            return build().collect()
        except pl.exceptions.ComputeError:
            if attempt == attempts:
                raise
            time.sleep(0.01 * attempt)
    raise AssertionError("unreachable")
