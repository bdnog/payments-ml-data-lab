# payments-ml-data-lab

![CI](https://github.com/OWNER/payments-ml-data-lab/actions/workflows/ci.yml/badge.svg)
<!-- Replace OWNER with the GitHub account or organisation once the repo is published. -->

A portfolio monorepo that models the two sides of a payments system with entirely synthetic data:
**`risk-scoring-api`**, an online fraud-scoring service (FastAPI, with a LightGBM baseline and a PyTorch
model planned), and **`settlement-service`**, a batch settlement and reconciliation pipeline (Polars)
with a read-mostly API that exposes results to BI. Both live in one uv workspace and share a small
library, with CI, Docker and Compose working from day one. The project is built in phases (see the
[roadmap](#roadmap)); right now it is the foundation: every service answers `GET /health`.

## Architecture (target)

```mermaid
flowchart LR
    client["Payment client"]
    bi["BI tool"]

    subgraph online["Online path"]
        risk["risk-scoring-api<br/>FastAPI"]
    end

    subgraph batch["Batch path"]
        job["settlement batch<br/>Polars"]
        marts[("Parquet marts")]
        sapi["settlement-service API<br/>read-mostly"]
        job -->|writes| marts
        sapi -->|reads| marts
    end

    pg[("Postgres")]
    rd[("Redis")]

    client -->|score request| risk
    risk --- rd
    risk --- pg
    job --- pg
    marts -->|Parquet| bi
    bi -->|queries| sapi
```

More detail in [`docs/architecture.md`](docs/architecture.md).

## Quickstart

Requirements: [uv](https://docs.astral.sh/uv/) and, for the stack, Docker with Compose. uv installs
Python 3.12 (pinned in `.python-version`) if you do not have it.

```bash
make sync     # install all workspace packages and dev tools
make test     # run the tests of every member
make lint     # ruff check + format check
make up       # build and start postgres, redis and both APIs
```

Check that the services are alive:

```bash
curl localhost:8001/health   # risk-scoring-api
# {"status":"ok","service":"risk-scoring-api","version":"0.1.0"}

curl localhost:8002/health   # settlement-service
# {"status":"ok","service":"settlement-service","version":"0.1.0"}
```

Stop everything with `make down`. Configuration comes from environment variables; copy
`.env.example` to `.env` to change the defaults (`.env` is never committed).

### Running the APIs without Docker

```bash
make run-risk             # http://localhost:8001
make run-settlement-api   # http://localhost:8002
```

### Running the settlement batch

The batch is a stub for now: it validates the date, logs the request and exits 0.

```bash
# locally
uv run python -m settlement.batch run --date 2026-11-10

# with Compose (profile "batch": not started by `make up`)
SETTLEMENT_RUN_DATE=2026-11-10 docker compose run --rm settlement-batch

# with the image directly: same image as the API, different command
docker build -f services/settlement-service/Dockerfile -t settlement-service .
docker run --rm settlement-service python -m settlement.batch run --date 2026-11-10
```

## Repository layout

```
.
├── packages/
│   └── shared/              # payments-shared: HealthResponse, configure_logging()
├── services/
│   ├── risk-scoring-api/    # online fraud scoring (api, serving, training)
│   └── settlement-service/  # batch + api around a shared domain (domain, batch, api, storage)
├── data/                    # synthetic data generators and small samples
├── contracts/               # OpenAPI exports, data contracts, payload examples
├── docs/
│   ├── architecture.md
│   └── adr/                 # architecture decision records
├── docker-compose.yml
├── Makefile
└── pyproject.toml           # uv workspace root + shared ruff/pytest config
```

Each service has its own `pyproject.toml` and `Dockerfile`; the build context for images is the repo
root because the workspace and the single `uv.lock` are needed.

## Decisions

- [ADR 0001: Monorepo with a uv workspace](docs/adr/0001-monorepo-with-uv-workspace.md)
- [ADR 0002: settlement-service has `batch` and `api` modules](docs/adr/0002-settlement-service-batch-and-api-modules.md)

## Roadmap

- [x] **Phase 1: Foundation.** Workspace, shared package, both service skeletons with `/health`,
  Dockerfiles, Compose, CI, Makefile, docs.
- [ ] **Phase 2: Data generators and contracts.** Synthetic transaction generators, Parquet samples,
  data contracts and exported OpenAPI documents.
- [ ] **Phase 3: risk-scoring-api.** Features, LightGBM baseline, PyTorch model, online scoring endpoint.
- [ ] **Phase 4: settlement-service.** Domain rules, Polars batch pipeline, Parquet marts, read API for BI.
- [ ] **Phase 5: Polish.** Observability, performance notes, documentation pass, demo script.

## License

No license has been chosen yet.
