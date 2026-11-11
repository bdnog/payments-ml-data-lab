# 0011. Hive-partitioned Parquet marts, atomic partition overwrite, empty partitions, exports

- Status: accepted
- Date: 2026-11-11

## Context

The marts are consumed by an API and, directly, by BI tools. They are rewritten when a run is repeated,
possibly while someone is reading them. A reader must never see a half-written file or a date that
silently disappears, and a BI tool should be able to query them with no code of ours.

## Decision

**Layout.** `<marts_dir>/<mart>/settlement_date=YYYY-MM-DD/part-0.parquet`, zstd, rows sorted by the
mart's primary key. The `settlement_date=` path segment is the Hive convention, so DuckDB, Polars,
Spark and BI connectors discover it as a column and skip excluded partitions. The column is also kept
inside the file, so an exported file is self-describing (both Polars and DuckDB accept the duplicate).

**Atomic partition overwrite.** To replace a partition we write the file to a staging directory under
`<marts_dir>/.staging` (outside every mart, so a glob of a mart never sees it), read it back and
validate it against the mart's data contract, then `os.replace` it onto the final path. A file rename
is atomic on one filesystem, so a reader sees the whole old file or the whole new one. Any failure (a
validation error, a full disk, a failed rename) leaves the previous partition untouched and removes the
staging directory; tests inject each of these failures. This is why a partition is a **single file**
rather than a directory of files: replacing a non-empty directory is not atomic, replacing a file is.
It deviates from "write to a temp directory and swap directories" for that reason. Rerunning a date
gives equal frames (and equal bytes), so a run is idempotent.

**Validation before publishing.** Every mart is validated against its data contract (types, nulls,
constraints, the `net = gross - fee - refunds - chargebacks` rule, the break-type values) and a
failure fails the run. Input validation can be skipped (`--skip-input-validation`); output
validation cannot, and a test shows that skipping the first still fails on the second.

**Empty partitions.** A date with zero payments or zero breaks still gets a partition: an empty file
with the right schema. "The run happened and found nothing" is thereby distinguishable from "the run
never happened", which matters for anyone asking "was 2026-01-12 reconciled?". The export of the
former is a `200` with an empty file; of the latter a `404`.

**Break ids** are `brk_` + 4 hex digits of the day number + 4 of the sequence within the day:
deterministic (a rerun gives the same ids), unique across dates, sorted like the rows, and free of any
hash that could collide.

**Exports** serve the partition file with `FileResponse` and the media type
`application/vnd.apache.parquet`. The two path parameters are an enum and a parsed date, never
concatenated user text, so a traversal attempt is rejected before it touches the filesystem.

**Reads.** The API scans lazily with the date range pushed down to the partitions (verified with the
query plan: only the requested partition is opened) and uses keyset pagination.

## Consequences

- BI tools can query the marts directly (`docs/bi-access.md`, whose queries run in a test).
- **A caveat found by measurement:** Polars opens a local Parquet file more than once while reading
  it (the footer, then the data), so a Polars read that overlaps a swap can fail with a `ComputeError`
  (about one read in four against a tight rewrite loop). A reader that opens the file once (DuckDB,
  `open`, an HTTP download) is unaffected. The API wraps its reads in `collect_with_retry`, which
  rebuilds and retries the query: correct because the error is transient and the next attempt sees a
  whole file. `FileResponse` stats and then opens the file, so a swap in that microsecond gap could
  confuse a download; it has not been seen and a retry by the client is safe.
- Many small files: 180 days give 540 partitions of a few KiB each (2.9 MiB in total), which is fine
  here and would need compaction at scale.
- Concurrent writers to the same partition are not coordinated beyond the atomic swap: the last one
  wins. In normal operation the single worker is the only writer, and the runs table allows one live
  run per date.
- A crash can leave a staging directory behind; it is outside the marts and can be deleted.
