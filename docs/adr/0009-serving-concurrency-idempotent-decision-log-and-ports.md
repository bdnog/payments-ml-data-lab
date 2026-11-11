# 0009. Serving: threads for CPU-bound inference, an idempotent decision log, ports and adapters

- Status: accepted
- Date: 2026-11-11

## Context

Scoring is CPU-bound and blocking: NumPy, a torch forward pass and a synchronous database driver. A
payments core retries on timeouts, so the same transaction can arrive twice, sometimes at the same
moment. And scoring needs a durable record of what was decided.

## Decision

**Plain `def` handlers, not `async def`.** FastAPI runs a `def` handler in a threadpool, so the event
loop keeps accepting connections while the model computes. An `async def` handler that calls torch
would run the computation *on* the event loop and stall every other request for its duration (the
classic mistake: `async` is for waiting on I/O, not for computing). We set
`torch.set_num_threads` from settings (default 1) and use `torch.inference_mode()`. To use more
cores we **add uvicorn workers** (`WEB_CONCURRENCY`), each a separate process with its own copy of the
model and one torch thread, rather than more threads in one process: it avoids the GIL for the Python
parts of the pipeline and oversubscribing cores with torch's own thread pool. The cost is memory
(one model copy per worker; the model is 27 KB, the torch import is the real cost).

**Ports and adapters.** `DecisionRepository` is a `Protocol` (the port). `InMemoryDecisionRepository`
serves unit tests; `SqlDecisionRepository` (SQLAlchemy 2.0, sync `Session` per request through
`Depends`) serves Postgres. The same contract tests run against both, including a concurrency test,
so the in-memory adapter is a trustworthy stand-in. Migrations are Alembic, in
`services/risk-scoring-api/migrations/`, applied by a one-shot `risk-migrate` Compose service before
the API starts.

**Idempotent decision log, insert first.** The `decisions` table has `transaction_id` as primary key.
Each request is scored (deterministic), then `INSERT ... ON CONFLICT (transaction_id) DO NOTHING
RETURNING`. If nothing was inserted, the existing row is loaded and its `request_hash` (sha256 of the
canonical JSON, sorted keys) is compared: equal means a replay, and the **stored** response is
returned (`200`, same body including `latency_ms` and `model_version`, even if a newer model is now
live); different means `409` problem+json. Insert-first avoids the race of "select, then insert":
two identical requests at once both compute the same answer, one wins the insert, and the other
blocks on the unique index until it commits and then reads the winner. A test with 12 threads
checks one row and one answer, and was shown to fail against a select-then-insert implementation.

**Failure behavior.** If the model failed to load (missing file, hash mismatch, feature columns that
differ from the contract) the app still starts, `/ready` answers `503` and the scoring routes answer
`503`; `/health` stays a pure liveness probe. If the decision log is unreachable the API answers
`503` (`urn:problem:decision-log-unavailable`): we do not score without recording, because the
record is what makes a retry safe. The caller decides the fallback (see `docs/risk-scoring.md`).

**One database per service.** Compose creates `risk_scoring` and `settlement` databases on first
start; the services never share tables. The risk image now includes `contracts/data`, because the
feature pipeline reads the data contract at runtime (this partly revisits ADR 0004).

**Model files.** Weights are a torch `state_dict` loaded with `weights_only=True`, and the loader
verifies the sha256 of `model.pt` and `preprocessor.json` recorded in `metadata.json` before using
them. This is a small supply-chain control, not a signature.

## Consequences

- Throughput scales with worker count; latency per request is the model cost (about 2 ms in process).
- Each decision costs one database round trip; with the database down the API cannot score. That is
  a deliberate choice of consistency over availability for the log, and shifts the fallback to
  the caller.
- A conflict is detected only for ids already in the log; there is no expiry, so the table grows
  with traffic (retention is out of scope).
- Because replays return the stored row, a retry after a model upgrade gets the old decision. That is
  the intended meaning of idempotent here.
- The in-memory adapter and the SQL adapter must stay behaviorally identical; the shared contract
  tests are what enforce it.
