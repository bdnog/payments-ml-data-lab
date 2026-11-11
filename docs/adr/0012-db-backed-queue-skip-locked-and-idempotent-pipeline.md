# 0012. A DB-backed queue with SKIP LOCKED, an API/worker split, and an idempotent pipeline

- Status: accepted
- Date: 2026-11-11

## Context

A settlement run for a large date can take seconds to minutes and must not block an HTTP request. It
must survive a restart, never run twice at once for the same date, and be safe to retry. Airflow, a
message broker or a task framework would do all this, but they are far more infrastructure than one
table needs, and are out of scope.

## Decision

**The API registers; a worker executes.** `POST /v1/runs` inserts a `PENDING` row and returns `202`
with a `Location` header; the worker, a separate process of the same image
(`python -m settlement.batch worker`), claims and runs it. The API is the only reader of the marts
volume (mounted read-only) and the worker the only writer.

**Idempotent by date, in the database.** A partial unique index (`UNIQUE (run_date) WHERE status <>
'FAILED'`) allows one live run per date. `POST` inserts with `ON CONFLICT ... DO NOTHING`; if nothing
was inserted it returns the existing run with `200`. A `FAILED` run is history and does not block a
new request. This holds under concurrency (a test with ten threads creates one run).

**Claiming with `FOR UPDATE SKIP LOCKED`.** `UPDATE runs SET status='RUNNING', attempts=attempts+1, ...
WHERE run_id = (SELECT run_id ... WHERE status='PENDING' ORDER BY requested_at FOR UPDATE SKIP LOCKED
LIMIT 1) RETURNING *` takes the oldest pending run in one statement. Rows another worker holds are
skipped, not waited on, so any number of workers poll the same table without blocking each other (a
test holds the oldest row locked from another connection and checks the claim returns the next one
immediately; it fails without `SKIP LOCKED`). Two workers draining 24 runs execute each exactly once.

**Valid transitions only**, enforced by a pure state machine in `domain/` and, in SQL, by the `WHERE`
clause of each guarded `UPDATE`: `PENDING -> RUNNING -> DONE | FAILED`, and `RUNNING -> PENDING` when a
stale run is given back. A stale `RUNNING` run (no result within `stale_after_seconds`) is re-queued, or
failed with a clear message once `attempts >= max_attempts`.

**At-least-once delivery, made safe by an idempotent pipeline.** If a worker dies mid-run, its run
runs again: that is the *at-least-once* guarantee, and there is deliberately no attempt at exactly-once
(which needs distributed transactions). It is safe because the pipeline is idempotent: rerunning a date
overwrites its three partitions atomically and gives the same data (ADR 0011). A worker slower than
`stale_after_seconds` can therefore run twice at the same time; `complete` and `fail` are guarded by the
**attempt number**, so the older attempt cannot overwrite the newer one's result.

**Graceful shutdown.** SIGTERM and SIGINT only set a flag: the run in progress finishes and is
recorded, then the worker exits without claiming another (`docker stop` exits 0 at once on an idle
worker; Compose gives a 2-minute grace period for a busy one).

**Ports and adapters**, as in the risk service: a `RunRepository` protocol with in-memory and SQL
adapters, and one contract-test suite that runs on both. Migrations are Alembic; a one-shot
`settlement-migrate` Compose service applies them before the API and worker start.

**The CLI `run` and `run-range` bypass the table**: an ops and backfill path, idempotent through the
partition overwrite.

## Consequences

- No new infrastructure: the queue is a table in the Postgres we already run, and the worker is a
  loop. Throughput is bounded by the pipeline, not by the queue.
- Polling (default every 2 seconds) adds latency a push-based broker would not; `LISTEN/NOTIFY` could
  remove it later.
- A worker has no heartbeat: a legitimately long run is indistinguishable from a dead one after
  `stale_after_seconds` (default 15 minutes), and would run twice (safely).
- `SKIP LOCKED` is a PostgreSQL (and some other databases) feature; the in-memory adapter has the same
  semantics only for tests.
- The queue gives no priorities, retries with backoff, or scheduling: a scheduler (cron, or an Airflow
  DAG calling `POST /v1/runs`) would sit in front of it. That is a natural Phase 5 extension.
