# Prompt: Phase 4 of `payments-ml-data-lab` (settlement-service for real, Polars)

## Role and context

You are continuing work on the **public portfolio monorepo** `payments-ml-data-lab`. Phases 1 to 3 are done: uv workspace, `packages/shared` (data-contract loader and validator), `packages/datagen` (synthetic universe with injected reconciliation breaks), YAML data contracts in `contracts/data/` (including the three mart contracts, so far without a producer), a **working** `risk-scoring-api` (use it as the reference for conventions: ports and adapters, SQLAlchemy 2.0 + Alembic, problem+json errors, `/ready`, Makefile and Compose patterns), and a **contract-first** `settlement-service` (Pydantic schemas, stub routes returning `501`, exported OpenAPI, `batch` CLI stub, `domain/` and `storage/` placeholders). I am a senior Java/Spring Boot engineer learning Python and data engineering, so keep the code idiomatic, typed and readable, and explain non-obvious choices in ADRs or short docstrings.

**Phase 4 turns `settlement-service` into a working system:**
1. **Batch pipeline** (Polars, lazy where it makes sense): compute the ledger's expected settlement, **reconcile it against the PSP file**, and publish three marts as hive-partitioned Parquet.
2. **Verification** against the generator's ground truth (`injected_breaks`), which the pipeline itself must never read.
3. **Runs as a DB-backed queue**: the API registers a run, a **worker** executes it (`FOR UPDATE SKIP LOCKED`), idempotent by date.
4. **Read API** for BI: paginated settlements and breaks, Parquet exports, runs.

**Out of scope:** Airflow, dbt, Spark, Kafka, S3, FX or multiple currencies, payout batching, late-arriving or cross-date mismatched records, a double-entry ledger journal, slowly changing merchant terms (terms are static), Prometheus metrics.

## Ground rules

- Read the repo first (`README.md`, `Makefile`, root and service `pyproject.toml`, ADRs, `contracts/`, `docs/data-model.md`, `packages/datagen/src/payments_datagen/settlement.py`, `services/risk-scoring-api` for conventions) and **follow existing conventions**. If something differs from what this prompt assumes, adapt and say so in the final report.
- All repo content in **English**. Do not ask me questions; decide, and record non-obvious decisions in ADRs.
- **The Phase 2 contracts are the source of truth** (data contracts and the OpenAPI). Implement them. The only intended contract changes are listed in this prompt. `make contracts-check` must pass after regenerating, and every contract change must be listed in the final report.
- **Where conventions are ambiguous** (for example the settlement date of REFUND and CHARGEBACK records, or what `injected_breaks.payment_id` holds for orphan PSP records): the **generator is the reference for what the data means**. Mirror its behavior in the pipeline and document it in `docs/settlement.md`. Change the generator only if the ground truth cannot be derived from the observable input data (otherwise the verification would be unfair), and list every generator change; if you change it, regenerate `data/samples/` and keep all Phase 2 tests passing.
- **The pipeline must never read `injected_breaks`.** Only the verification module and tests may. A test must prove the pipeline works with that file deleted.
- Money is integer minor units everywhere. No floats for amounts, ever.
- Allowed runtime dependencies: `polars`, `sqlalchemy>=2`, `alembic`, `psycopg[binary]` (v3). Dev-only: `duckdb` (to prove that BI tools can query the marts). Not allowed: pandas, pyarrow, airflow, dbt, pyspark, kafka clients.
- One Conventional Commit per step. Do not push. Never claim something works without running it; if a check cannot run, say so in the final report.

## Decisions already made

- **Domain core** (`settlement/domain/`, no Polars, FastAPI or SQLAlchemy imports): enums (`RecordType`, `BreakType`, `RunStatus`), `expected_fee_minor(gross, mdr_bps)` with **integer round-half-up** (`(gross * mdr_bps + 5000) // 10000`), `settlement_date(captured_at, settlement_days)` (UTC calendar date of capture plus N days), constants such as the flat chargeback fee. The batch layer has Polars **expression equivalents** of these two functions, and a **parity test** over a grid of values proves expression and scalar agree.
- **Run semantics**: a run for date `D` reconciles everything **due on `D`**. Pipeline functions are **date-agnostic** (they group by `settlement_date`, so one pass can handle many dates). `run --date D` is a list of one date; `run-range --from --to` handles many dates in a single pass and writes each partition separately.
- **Expected ledger side** for date `D`: SALE for each `CAPTURED` payment whose due date is `D`; REFUND and CHARGEBACK expected records per the generator's rules. PSP side: records with `settlement_date = D`. Matching key: `(payment_id, record_type)`, where the PSP's `merchant_reference` is the payment id.
- **Break classification** (complete: there is no "unclassified" state), with the amounts that populate the `reconciliation_breaks` mart:

  | Situation | break_type | ledger_amount_minor | psp_amount_minor | diff_minor |
  |---|---|---|---|---|
  | More than one PSP record per key (all but the lowest `psp_record_id` are duplicates) | DUPLICATE_IN_PSP | null | duplicate's gross | +duplicate gross |
  | Ledger expected, no PSP record | MISSING_IN_PSP | ledger amount | null | -ledger amount |
  | PSP record with no expected ledger counterpart, type CHARGEBACK | CHARGEBACK_NOT_IN_LEDGER | null | psp gross | +psp gross |
  | PSP record with no expected ledger counterpart, other types | MISSING_IN_LEDGER | null | psp gross | +psp gross |
  | Both present, gross differs | AMOUNT_MISMATCH | ledger amount | psp gross | psp - ledger |
  | Both present, gross equal, SALE, fee differs from the expected fee | FEE_MISMATCH | expected fee | psp fee | psp - expected |

  AMOUNT_MISMATCH takes **precedence** over FEE_MISMATCH. The expected fee uses the ledger amount and the merchant's `mdr_bps`. Document the known limitation: a record settled on a different date than expected appears as MISSING_IN_PSP on one date and MISSING_IN_LEDGER on another.
- **What the marts mean**: `daily_settlement_by_merchant` is the **ledger-expected payout** ("what we owe the merchant"), not what the PSP confirmed: `gross` = captured payment amounts due that day, `fee` = expected fee, `refunds`, `chargebacks` (amounts only, the flat chargeback fee is absorbed by the platform), `net = gross - fee - refunds - chargebacks` (may be negative). All components are non-negative magnitudes. Add `net_minor = gross_minor - fee_minor - refunds_minor - chargebacks_minor` as a **row rule** in the mart contract if it is not already there (a data-contract change to list). `fee_summary` covers only SALEs matched on both sides **with equal gross** (so a non-zero `fee_diff_minor` isolates FEE_MISMATCH), with `actual_fee` from the PSP and `fee_diff = actual - expected`. `reconciliation_breaks` carries `run_date` (the date being reconciled) and `settlement_date` (the same date; keep the column as the contract defines it).
- **Mart storage**: `<marts_dir>/<mart>/settlement_date=YYYY-MM-DD/part-0.parquet` (zstd), rows sorted by the mart's primary key. **Atomic partition overwrite**: write to a temp directory beside the target, validate against the mart contract, then swap into place, so a rerun is idempotent and a failure never leaves a partial or missing partition. A run that finds zero breaks (or zero payments) **still writes an empty partition with the correct schema**, so "run happened, nothing found" is distinguishable from "never run". Document this in an ADR.
- **Runs**: Postgres `runs` table via SQLAlchemy 2.0 and Alembic (the `settlement` database already exists from Phase 3): `run_id` (format as the contract defines, default `run_` + 12 hex), `run_date`, `status` (PENDING, RUNNING, DONE, FAILED), `requested_at`, `started_at`, `finished_at`, `error` (truncated to 2,000 chars), plus internal columns not exposed by the API: `attempts`, `claimed_at`, `summary` (JSONB). A **partial unique index** allows only one non-FAILED run per `run_date`. Ports and adapters as in Phase 3: a `RunRepository` protocol with in-memory and SQL adapters, sharing one contract-test suite. Valid transitions only: PENDING to RUNNING to DONE or FAILED; stale RUNNING goes back to PENDING (or FAILED when `attempts >= max_attempts`).
- **Worker** (`python -m settlement.batch worker [--once]`): claims with `UPDATE ... WHERE id = (SELECT id ... WHERE status='PENDING' ORDER BY requested_at FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING ...`, runs the pipeline, records DONE or FAILED with the summary, sleeps `worker_poll_seconds` when idle, recovers stale runs (`stale_after_seconds`), and shuts down gracefully on SIGTERM (finish the current run, then exit). Delivery is at-least-once, safe because the pipeline is idempotent; say so in the ADR.
- **CLI direct paths** (`run`, `run-range`) bypass the runs table: an ops and backfill path, idempotent through partition overwrite.
- **Read API**: plain `def` endpoints (Polars is blocking). Reads use `scan_parquet` with hive partitioning and **filter pushdown on `settlement_date`**. Keyset cursor pagination, ordering `(settlement_date, merchant_id)` for settlements and `(settlement_date, break_id)` for breaks; the cursor is an opaque base64url JSON of the last key, and a tampered or malformed cursor returns a `400` problem+json (`urn:problem:invalid-cursor`, declare it in the OpenAPI). Defaults: `limit` 100, maximum 1,000. The `date` filter of `/v1/reconciliation/breaks` is the `settlement_date`, unless the Phase 2 OpenAPI says otherwise. Exports serve the partition file with `FileResponse` and media type `application/vnd.apache.parquet`, `404` problem+json if the partition does not exist, and no user-controlled path concatenation (mart is an enum, date is parsed). `GET /ready` checks DB connectivity and that the marts directory is readable; add it to the OpenAPI. `POST /v1/runs` returns `202` with a `Location` header for a new run and `200` with the existing run when the date already has a non-failed one.
- **Input validation**: at the start of a run, validate the input datasets against their data contracts and check foreign keys (`payments.merchant_id` in merchants, and so on), with a `--skip-input-validation` flag. **Validate each output mart against its contract before publishing.** A validation failure fails the run.

## Target structure (additions and changes)

```
services/settlement-service/
├── alembic.ini
├── migrations/versions/0001_create_runs.py
├── src/settlement/
│   ├── config.py                    # + database_url, input_dir, marts_dir, worker settings
│   ├── domain/                      # enums, fee rule, settlement date, constants (pure Python)
│   ├── storage/
│   │   ├── datasets.py              # lazy readers for the input layout + contract validation
│   │   ├── marts.py                 # atomic hive-partition writer, partition reader, empty partitions
│   │   └── cursors.py               # encode/decode keyset cursors
│   ├── batch/
│   │   ├── __main__.py              # CLI: run, run-range, worker, verify
│   │   ├── expected.py              # ledger-side expected records (LazyFrame)
│   │   ├── reconcile.py             # classification per the table above
│   │   ├── marts.py                 # daily_settlement_by_merchant, fee_summary builders
│   │   ├── pipeline.py              # run_dates(dates, input_dir, marts_dir) -> RunSummary
│   │   ├── worker.py                # claim loop, graceful shutdown, stale recovery
│   │   └── verification.py          # compares breaks mart with injected_breaks (never imported by the pipeline)
│   ├── persistence/                 # models.py, repository.py (protocol + adapters), session.py
│   └── api/                         # lifespan, deps, errors (problem+json), routes
└── tests/                           # domain, storage, batch, persistence, api, e2e
scripts/demo_settlement.py           # posts a run, polls, prints pages, downloads an export and reads it with Polars
docs/
├── settlement.md                    # rules, taxonomy, amounts table, mart semantics, run lifecycle (Mermaid sequence)
├── bi-access.md                     # querying the marts directly with DuckDB (hive partitioning)
└── adr/0010..0012
```

Input layout read by the pipeline (the Phase 2 dataset root): `reference/merchants.parquet`, `settlement/{payments,refunds,chargebacks,psp_settlement}.parquet`. Defaults: `SETTLEMENT_INPUT_DIR` and `SETTLEMENT_MARTS_DIR` (the Makefile passes absolute paths; the image uses `/data/input` and `/data/marts`). `data/marts/` is gitignored.

## Verification module

`verify` compares the `reconciliation_breaks` mart with `settlement/injected_breaks.parquet` over the dates covered by the marts, matching on `(break_type, payment_id, psp_record_id)` with nulls equal, and also compares `diff_minor` with `expected_diff_minor`. It prints precision and recall **per break type**, the first few mismatches, and exits non-zero unless both are 100%. For DUPLICATE_IN_PSP the pipeline's rule is "lowest `psp_record_id` is the original"; if the generator flags the other copy, align the generator (the duplicate gets the higher id) and list that change.

## Testing requirements

- **Domain**: fee rounding table including exact halves (for example gross 1050 at 100 bps is 11, gross 1049 is 10), settlement date around midnight UTC, enum values equal the data-contract `accepted_values`.
- **Parity**: Polars expressions equal the domain scalar functions over a grid of gross and mdr values and capture timestamps.
- **Reconcile** on small hand-built frames: one case per break type, a clean match, a not-yet-due payment (no break), AMOUNT_MISMATCH precedence over FEE_MISMATCH, duplicates keep the lowest id, refunds and chargebacks matched and unmatched.
- **Marts**: row rule `net = gross - fee - refunds - chargebacks`; totals tie out to the ledger (sum of `gross_minor` equals the sum of due CAPTURED payment amounts); `fee_summary` difference per day equals the sum of FEE_MISMATCH diffs; every mart validates against its contract.
- **Storage**: atomic overwrite (inject a failure mid-write: the previous partition is intact and no temp dir is left), idempotent rerun (equal frames), empty partitions written with the right schema, partitions discoverable by Polars hive scan **and by DuckDB** (dev test running SQL over `read_parquet(..., hive_partitioning=true)`).
- **End to end**: on the committed sample **and** on small datasets generated with three different seeds through `payments_datagen`, `run-range` then `verify` gives exactly 100% precision and recall for every break type; the pipeline also works with `injected_breaks.parquet` deleted from a temp copy of the input.
- **Persistence**: repository contract tests over both adapters (the SQL one skipped with an explicit reason unless `SETTLEMENT_TEST_DATABASE_URL` is set): partial unique index (a second non-failed run for a date is rejected, a failed one allows a new run), illegal transitions rejected, **concurrency**: two worker threads and N pending runs, each run executed exactly once; stale recovery and the `max_attempts` cutoff.
- **Worker**: `--once` drains the queue, FAILED is recorded with the error when the pipeline raises, SIGTERM-style stop finishes the current run.
- **API** (`TestClient`, in-memory repository, a small marts directory built by the pipeline): walking all pages yields every row exactly once in stable order; a tampered cursor returns `400`; filters; limit bounds; export `404` and path-traversal attempts rejected; `POST /v1/runs` returns `202` then `200` with the same `run_id`; `GET /v1/runs/{id}`; `/ready` states.
- Phase 2 contract tests (examples, alignment of page models with mart contracts, OpenAPI check) keep passing; regenerate the OpenAPI and review the diff.

## Step-by-step plan (one commit per step)

1. **Dependencies and config**: dev `duckdb`, settings, `.gitignore` entry for `data/marts/`; verify `uv sync`, Docker build and CI still work.
2. **Rules extraction and domain**: read the generator, write the rules section of `docs/settlement.md` (settlement dates for each record type, fee rule, sign semantics, due-date cutoff, reference conventions) flagging any mismatch with the Phase 2 text, then implement the domain and its tests.
3. **Storage layer**: lazy dataset readers with contract and FK validation, the atomic mart writer and reader, cursors, with tests including failure injection.
4. **Pipeline**: expected records, reconciliation, the two mart builders, the parity tests and the hand-built scenarios.
5. **CLI and verification**: `run`, `run-range`, `verify`; end-to-end tests (sample, three seeds, ground truth deleted, DuckDB); mart contract rule for `net` if needed.
6. **Runs persistence**: SQLAlchemy model, Alembic migration, repository protocol and adapters, state machine, tests including concurrency.
7. **Worker**: claim loop, graceful shutdown, stale recovery, `--once`, tests.
8. **Read API**: lifespan, `/ready`, settlements, breaks, exports, runs, cursor pagination, problem+json handlers, tests; regenerate the OpenAPI (`make contracts`) and review the diff.
9. **Docker, Compose, CI**: the image includes `migrations/` and `alembic.ini` and a writable `/data/marts` owned by the non-root user; Compose gets a one-shot `settlement-migrate`, a `settlement-worker` service (same image, command `python -m settlement.batch worker`), a shared named volume for marts (read-write for the worker, read-only for the API), the dataset directory mounted read-only from `DATASET_DIR` (default `./data/samples`), and a healthcheck on `/ready`. The existing `settlement-batch` profile service stays for ad-hoc runs. CI: add the settlement test database URL to the Postgres service and a smoke step (`run-range` on the sample into a temp dir, then `verify`).
10. **Full-data run and DX**: Makefile targets `settle-run DATE=`, `settle-all DATA=`, `verify-settlement`, `migrate-settlement`, `demo-settlement`, `bench-settlement`; `scripts/demo_settlement.py`. Run everything on `data/out/full` (generate it with the Phase 2 command if missing) and collect the numbers.
11. **Docs**: finish `docs/settlement.md` (Mermaid sequence diagram of API, DB, worker and Parquet, plus the amounts table and mart semantics), `docs/bi-access.md` (DuckDB queries a BI user would run, verified by the dev test), the marts section of `docs/data-model.md`, ADRs 0010 (reconciliation algorithm, taxonomy and ground truth kept separate), 0011 (hive-partitioned Parquet marts, atomic partition overwrite, empty partitions, exports), 0012 (DB-backed queue with SKIP LOCKED, API and worker split, at-least-once plus idempotent pipeline), README (settlement quickstart, a results table with the real numbers, roadmap with Phase 4 checked).

## Definition of done

Run these yourself and show the results:

1. `uv sync --all-packages`, `uv run ruff check .`, `uv run ruff format --check .` and `uv run pytest` pass; the SQL tests also pass with `make test-db`.
2. `make settle-all DATA=data/out/full` completes. Report wall-clock time (single date and the whole range), rows per mart, breaks by type, and the total expected payout. Then `make verify-settlement` shows **100% precision and recall for every break type**. If not, investigate the generator-versus-pipeline convention before anything else, and report it.
3. `make contracts-check` passes; list every contract change (the `/ready` route, the `Location` header, the `400` invalid-cursor response, the `net` rule if added, and anything else).
4. `docker compose up -d` brings up Postgres, migration, `settlement-service` API and the worker. Then `make demo-settlement` shows: `POST /v1/runs` returns `202`, polling reaches `DONE`, settlements and breaks pages return data, an export downloads and reads back with Polars, and posting the same date again returns `200` with the same `run_id`.
5. `make bench-settlement` reports the time to process one date and the whole range on the full dataset, and the size of the marts on disk.
6. `docker compose build` works; report the settlement image size.
7. `git log --oneline` shows small conventional commits, and `git status` is clean.

## Final report format

Keep it short: what was created (tree), the results of the seven checks (pass, fail or not run, with reasons), a table with the reconciliation results per break type and the main performance numbers, every contract and generator change, any decision not obvious from this prompt, and suggested next steps for Phase 5 (polish: load tests, README with diagrams, optionally an Airflow DAG that calls the runs API, the Spring Boot payments-core stub, and a double-entry ledger journal).
