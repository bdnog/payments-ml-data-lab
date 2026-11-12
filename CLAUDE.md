# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A uv-workspace monorepo (Python 3.12) with entirely synthetic data. Two services plus two library packages:

- `packages/shared` (`payments_shared`): schemas, logging, RFC 7807 problem helpers, the YAML data-contract loader/validator. Keep it small; domain logic belongs to the owning service.
- `packages/datagen` (`payments_datagen`): deterministic generator (seeded) of one universe: labeled transactions for risk, plus ledger/PSP file with injected reconciliation breaks for settlement.
- `services/risk-scoring-api` (`risk_scoring`, port 8001): online fraud scoring. `training/` (offline, PyTorch MLP) and `serving/` + `api/` (online) share one feature pipeline; the promoted model in `models/current/` is committed and hash-verified at load.
- `services/settlement-service` (`settlement`, port 8002): `batch/` (Polars reconciliation, worker, verification), `domain/` (pure rules, no Polars/FastAPI/storage imports), `storage/` + Parquet marts (the contract between batch and API), `api/` (read-mostly), `persistence/` (Postgres run queue).

Phases 1 to 4 are done; Phase 5 (observability, polish) is open. Redis is provisioned but unused.

## Commands

Everything goes through `uv` and the Makefile (`make help` lists targets).

```bash
make sync                      # uv sync --all-packages
make test                      # uv run pytest (all four members; Postgres tests skip without a DB URL)
make lint / make format        # ruff check + format check / auto-fix (line length 100)
make contracts-check           # regenerate OpenAPI and fail on git diff (CI runs this)
make test-db                   # starts Postgres via Compose, runs the SQL/persistence tests on throwaway DBs
make up / make down            # Compose stack: postgres, redis, migrations, both APIs, settlement worker
```

Single test: `uv run pytest services/settlement-service/tests/path/test_x.py::test_name` (pytest uses `--import-mode=importlib`, so test file basenames need not be unique).

SQL tests only run with `RISK_TEST_DATABASE_URL` / `SETTLEMENT_TEST_DATABASE_URL` set (see `SKIP_SQL` in the `persistence/conftest.py` files); `make test-db` sets them. Each service migrates its own database with Alembic (`make migrate-risk`, `make migrate-settlement`).

Data and pipelines:

```bash
make data-sample                         # regenerate committed data/samples (deterministic, seed 42)
make data-full                           # ~500k txns into data/out/full (git-ignored)
make data-validate DIR=data/out/full
make train-risk VERSION=mlp-2026.11.1    # needs data-full; then make promote-model VERSION=...
make settle-all DATA=data/out/full       # reconcile every date in the PSP file
make verify-settlement DATA=data/out/full  # must report 100% vs the answer key or exits non-zero
make settle-run DATE=2026-01-15          # one date, bypassing the queue
make run-risk / make run-settlement-api  # local uvicorn with reload (risk needs Postgres + migrate-risk)
```

## Architecture notes that span files

- **Contract-first.** YAML data contracts in `contracts/data/` are the source of truth for datasets *and* the three settlement marts. OpenAPI JSON in `contracts/openapi/` is generated from the FastAPI apps (`scripts/export_openapi.py`); after changing any route or schema run `make contracts` and commit the result, or CI fails. Example payloads in `contracts/examples/` are validated by tests.
- **Money is integer minor units** (centavos, ADR 0005). Never use floats for amounts. Fee is `(gross * mdr_bps + 5000) // 10000` (round half up). In the PSP file `gross_amount_minor` is a positive magnitude and the sign lives in `net_amount_minor`; a break's `diff_minor` is always PSP minus ledger.
- **Settlement dates use the UTC calendar date** of the timestamp plus the merchant's `settlement_days` (refunds/chargebacks: +1 day). Reconciliation key is `(settlement_date, payment_id, record_type)`. Rules mirror the generator (`datagen/settlement.py`); the generator is the reference when a convention is ambiguous. See `docs/settlement.md`.
- **Ground truth stays apart (ADR 0010).** `settlement/batch/verification.py` is the only production module allowed to read `injected_breaks`; the pipeline must never import or depend on it (a test enforces this).
- **Marts are Hive-partitioned Parquet** written with atomic overwrite (ADR 0011); the API and BI/DuckDB read them directly. The run queue is Postgres `SKIP LOCKED` with an API/worker split and an idempotent pipeline (ADR 0012). The same Docker image runs API, worker and batch by overriding the command.
- **No training/serving skew.** Training and serving go through the same feature pipeline (ADR 0007); training uses label maturity and a temporal split (ADR 0008). Scoring takes precomputed behavior features (ADR 0006) and writes an idempotent decision log (ADR 0009).
- Errors are RFC 7807 with problem types `urn:problem:<slug>`.
- Docker build context is the repo root (single workspace + `uv.lock`). `torch` resolves to CPU-only wheels on Linux via `tool.uv.sources`.

Decisions are recorded in `docs/adr/`; read the relevant ADR before changing one of the behaviors above. Config comes from env vars (`.env.example`; `.env` is never committed). `phase*-prompt.md` and `scaffold-prompt.md` at the root are the original build briefs, not documentation to keep in sync.
