# Prompt: scaffold the `payments-ml-data-lab` monorepo

## Role and context

You are helping me bootstrap a **public portfolio monorepo** called `payments-ml-data-lab`. I am a senior Java/Spring Boot engineer learning Python (FastAPI and data engineering). The repo will eventually contain two services that model the two sides of a payments system:

- **risk-scoring-api**: online fraud scoring (FastAPI; PyTorch model and a LightGBM baseline will be added in a later phase).
- **settlement-service**: batch settlement and reconciliation pipeline (Polars) plus a read-mostly FastAPI exposing results to BI. It has two execution modes (`batch` and `api`) that share one `domain` package.

**This task is the skeleton only.** No business logic, no ML, no data generators. The goal is a repo where `uv sync`, lint, tests, CI and `docker compose up` all work from day one, with every service answering `GET /health`.

## Ground rules

- All repo content (code, comments, README, ADRs, commit messages) must be in **English**.
- Do not ask me questions. Make reasonable decisions, and record the non-obvious ones in ADRs.
- Do **not** add PyTorch, LightGBM, scikit-learn, pandas or DuckDB yet. They are heavy and arrive in later phases. Polars is allowed in `settlement-service` because it is light.
- Everything in the repo must be synthetic or generic. No real data, no secrets. Use `.env.example` for configuration, never a committed `.env`.
- Work in small steps and make **one Conventional Commit per step** (`chore:`, `feat:`, `docs:`, `ci:`, `build:`, `test:`). Run `git init` if needed. Do not create a remote and do not push.
- Never claim something works without running it. If a verification step cannot run (for example Docker is not available), say so explicitly in your final report instead of skipping silently.

## Tooling decisions (already made)

- Python **3.12**, pinned in `.python-version`.
- **uv** workspace (a single lockfile at the root, commit `uv.lock`).
- **ruff** for lint and format, **pytest** for tests, **pytest-cov** optional.
- Build backend: `hatchling`, with the `src/` layout in every member.
- **FastAPI**, **Pydantic v2**, **pydantic-settings**, **uvicorn**; **httpx** for tests (`TestClient`).
- Dependencies for the DB (SQLAlchemy, Alembic, psycopg) are **not** added yet, but Postgres and Redis run in Compose so later phases can use them.

## Target structure

```
payments-ml-data-lab/
├── .github/workflows/ci.yml
├── .gitignore
├── .python-version
├── .env.example
├── .dockerignore
├── Makefile
├── README.md
├── pyproject.toml                 # uv workspace root + shared ruff/pytest config
├── uv.lock
├── docker-compose.yml
├── packages/
│   └── shared/                    # distribution name: payments-shared, import: payments_shared
│       ├── pyproject.toml
│       ├── src/payments_shared/
│       │   ├── __init__.py
│       │   ├── logging.py         # configure_logging(): structured-ish stdlib logging
│       │   └── schemas.py         # HealthResponse (status, service, version)
│       └── tests/test_schemas.py
├── services/
│   ├── risk-scoring-api/          # distribution: risk-scoring-api, import: risk_scoring
│   │   ├── pyproject.toml
│   │   ├── Dockerfile
│   │   ├── src/risk_scoring/
│   │   │   ├── __init__.py
│   │   │   ├── config.py          # pydantic-settings Settings
│   │   │   ├── api/
│   │   │   │   ├── __init__.py
│   │   │   │   └── main.py        # create_app() + GET /health
│   │   │   ├── serving/__init__.py    # placeholder: online inference (later phase)
│   │   │   └── training/__init__.py   # placeholder: offline training (later phase)
│   │   └── tests/api/test_health.py
│   └── settlement-service/        # distribution: settlement-service, import: settlement
│       ├── pyproject.toml
│       ├── Dockerfile
│       ├── src/settlement/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── domain/__init__.py     # pure rules, no Polars/FastAPI imports
│       │   ├── batch/
│       │   │   ├── __init__.py
│       │   │   └── __main__.py        # CLI: `python -m settlement.batch run --date YYYY-MM-DD` (stub)
│       │   ├── api/
│       │   │   ├── __init__.py
│       │   │   └── main.py            # create_app() + GET /health
│       │   └── storage/__init__.py
│       └── tests/
│           ├── api/test_health.py
│           └── batch/test_cli.py
├── data/
│   ├── generators/.gitkeep
│   └── samples/.gitkeep
├── contracts/
│   └── README.md                  # explains what will live here (OpenAPI exports, data contracts, payload examples)
└── docs/
    ├── adr/
    │   ├── 0000-template.md
    │   ├── 0001-monorepo-with-uv-workspace.md
    │   └── 0002-settlement-service-batch-and-api-modules.md
    └── architecture.md            # overview + Mermaid diagram of the target system
```

## Step-by-step plan

Execute in this order, committing after each step.

### Step 1: repo foundation
- `git init`, `.gitignore` (Python, uv, `.venv`, `.env`, `__pycache__`, `.ruff_cache`, `.pytest_cache`, `.coverage`, `data/**/*.parquet` except `data/samples/`), `.python-version`, `.env.example`.
- Root `pyproject.toml`:
  - `[project]` named `payments-ml-data-lab` (virtual workspace root, not published), `requires-python = ">=3.12"`.
  - `[tool.uv.workspace] members = ["packages/*", "services/*"]`.
  - `[dependency-groups] dev = ["pytest", "pytest-cov", "httpx", "ruff"]`.
  - `[tool.ruff]`: line length 100, target py312; lint rules `E, F, I, B, UP, SIM, N`; format enabled.
  - `[tool.pytest.ini_options]`: `testpaths` covering all members, `addopts = "-q"`.

### Step 2: shared package
- `packages/shared` with `HealthResponse` (Pydantic v2) and `configure_logging()`.
- One unit test for `HealthResponse`.

### Step 3: risk-scoring-api skeleton
- Dependencies: `fastapi`, `uvicorn[standard]`, `pydantic-settings`, plus the workspace dependency `payments-shared` (`[tool.uv.sources] payments-shared = { workspace = true }`).
- `create_app()` factory returning a FastAPI app with title/version from settings; `GET /health` returns `HealthResponse(status="ok", service="risk-scoring-api", version=...)`.
- `config.py`: `Settings(BaseSettings)` with `env_prefix="RISK_"`, fields `app_version`, `log_level`.
- Test: `TestClient` hits `/health` and asserts the payload.

### Step 4: settlement-service skeleton
- Dependencies: same as above plus `polars`.
- `api/main.py` with `create_app()` and `GET /health`.
- `batch/__main__.py`: a minimal CLI (use `argparse`, no extra dependency) with a `run` subcommand that requires `--date` (ISO format, validated), logs "settlement run requested for <date> (not implemented yet)" and exits 0.
- `domain/` stays empty except `__init__.py` with a docstring stating the rule: **no imports from Polars, FastAPI or storage here**.
- Tests: `/health` test and CLI test (valid date exits 0; invalid date exits non-zero).

### Step 5: Dockerfiles
- One multi-stage Dockerfile per service, **build context is the repo root** (the workspace and lockfile are needed), e.g. `docker build -f services/risk-scoring-api/Dockerfile .`.
- Base `python:3.12-slim`; copy `uv` from the official `ghcr.io/astral-sh/uv` image; install with `uv sync --frozen --no-dev --package <service>`; run as a **non-root user**; `HEALTHCHECK` hitting `/health`.
- `settlement-service` image: default command runs the API (`uvicorn settlement.api.main:app`), and the same image must be able to run the batch via `python -m settlement.batch run --date ...`. Document both in comments and README.
- Add a root `.dockerignore`.

### Step 6: docker-compose
- Services: `postgres` (16, with a named volume and healthcheck), `redis` (7, healthcheck), `risk-scoring-api` (port 8001), `settlement-service-api` (port 8002), and a `settlement-batch` service under a Compose **profile** `batch` (not started by default).
- Configuration comes from environment variables with defaults in `.env.example`. Both APIs `depends_on` Postgres and Redis healthchecks (even though they do not use them yet).

### Step 7: CI
- `.github/workflows/ci.yml`, triggered on `push` and `pull_request`:
  - Job `lint-and-test`: checkout, `astral-sh/setup-uv` with caching, `uv sync --all-packages --frozen`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest`.
  - Job `docker-build`: builds both images (no push) with `docker/build-push-action` and the GitHub Actions cache.
- Pin actions to major versions.

### Step 8: developer experience
- `Makefile` with targets: `sync`, `lint`, `format`, `test`, `run-risk`, `run-settlement-api`, `up`, `down`, `build`, `help` (default, lists targets). Use `uv run` everywhere.

### Step 9: documentation
- `README.md`: project pitch (one paragraph), a Mermaid diagram of the target architecture (risk-scoring online path, settlement batch path, Postgres/Redis, BI consuming Parquet marts), quickstart (`make sync`, `make test`, `make up`, `curl` examples for both `/health` endpoints), repo layout, a roadmap checklist with phases: 1 Foundation (checked), 2 Data generators and contracts, 3 risk-scoring-api, 4 settlement-service, 5 Polish, and a CI badge placeholder.
- `docs/architecture.md`: expands the diagram and explains the two "batch vs online" modes per service.
- ADR 0001: why a monorepo with a uv workspace. ADR 0002: why `settlement-service` has `batch` and `api` modules, with `domain` as the shared core and storage (Parquet marts) as the contract between them. Use the template in `0000-template.md` (Context, Decision, Consequences).
- `contracts/README.md`: what will live there and the conventions (OpenAPI JSON exported from each app, data-contract docs for the Parquet marts, example payloads).

## Definition of done

Run these yourself and show the results in the final report:

1. `uv sync --all-packages` succeeds and `uv.lock` is committed.
2. `uv run ruff check .` and `uv run ruff format --check .` pass.
3. `uv run pytest` passes (all members).
4. `uv run python -m settlement.batch run --date 2026-11-10` exits 0, and an invalid date exits non-zero.
5. `docker compose config` validates. If Docker is available: `docker compose build`, then `docker compose up -d`, then `curl localhost:8001/health` and `curl localhost:8002/health` both return `{"status":"ok",...}`, then `docker compose down`.
6. `git log --oneline` shows the small, conventional commits from the steps above, and `git status` is clean.

## Final report format

Keep it short: a tree of what was created, the results of the six checks above (pass, fail or not run, with the reason), any decision you made that is not obvious from the plan, and the suggested next step (phase 2: data generators and contracts).
