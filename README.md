# payments-ml-data-lab

![CI](https://github.com/OWNER/payments-ml-data-lab/actions/workflows/ci.yml/badge.svg)
<!-- Replace OWNER with the GitHub account or organisation once the repo is published. -->

A portfolio monorepo that models the two sides of a payments system with entirely synthetic data:
**`risk-scoring-api`**, an online fraud-scoring service (FastAPI, a PyTorch MLP with embeddings, an
idempotent decision log in Postgres), and **`settlement-service`**, a batch settlement and
reconciliation pipeline (Polars) with a read-mostly API that exposes results to BI. Both live in one
uv workspace and share a small library, with CI, Docker and Compose working from day one. The project
is built in phases (see the [roadmap](#roadmap)). Today the risk-scoring service works end to end
(features, training, evaluation, serving, decision log); the settlement service still answers `501`
on its `/v1` routes; and a deterministic generator produces one synthetic universe for both: labeled
transactions for the risk side, and a ledger and PSP file with known injected breaks for the
settlement side.

> **Everything here is synthetic.** The model is trained on this repository's own generator, so its
> metrics say nothing about real-world fraud performance.

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
make up       # build and start postgres, redis, the migration and both APIs
```

An existing Postgres volume from before Phase 3 needs a one-time `docker compose down -v`, so the
init script can create the `risk_scoring` and `settlement` databases.

Check that the services are alive:

```bash
curl localhost:8001/ready    # risk-scoring-api: 200 once the model is loaded (/health is liveness only)
# {"status":"ready","model_version":"mlp-2026.11.1"}

curl localhost:8002/health   # settlement-service
# {"status":"ok","service":"settlement-service","version":"0.1.0"}
```

Stop everything with `make down`. Configuration comes from environment variables; copy
`.env.example` to `.env` to change the defaults (`.env` is never committed).

### Synthetic data

```bash
make data-sample                                  # refresh the committed sample in data/samples (~250 KB)
make data-full                                    # about 500k transactions into data/out/full (ignored by git)
make data-validate DIR=data/out/full              # contracts, foreign keys, cross-dataset invariants
```

`data-full` takes a couple of seconds. Every run with the same seed gives the same data. The flags
(`--seed`, `--days`, `--fraud-rate`, `--break-rate`, ...) are listed by
`uv run python -m payments_datagen generate --help`. What is in the data: [`docs/data-model.md`](docs/data-model.md)
and [`docs/fraud-patterns.md`](docs/fraud-patterns.md).

### Risk scoring: train, serve, try it

```bash
make train-risk VERSION=mlp-2026.11.1   # needs data/out/full (make data-full); a few seconds on a CPU
make promote-model VERSION=mlp-2026.11.1  # copy the run to services/risk-scoring-api/models/current
make bench-risk                         # in-process latency: 1,000 single requests, p50 / p95
make up && make demo-score              # scores sample rows, a batch, a replay and a 409
make test-db                            # SQL tests of the decision log, against a throwaway Postgres
```

`models/current/` is committed (about 50 KB), so the service works right after cloning. How a request
flows, how skew is prevented, the idempotency rules and what a caller should do on a timeout are in
[`docs/risk-scoring.md`](docs/risk-scoring.md); the model's intended use and limits are in the
[model card](docs/model-card.md).

**Results** of `mlp-2026.11.1` on the `full` profile (seed 42), time-based split, **synthetic data**:

| | train | validation | test |
| --- | --- | --- | --- |
| rows | 263,054 | 77,210 | 159,402 |
| label rate (prevalence = PR-AUC of a random classifier) | 0.485% | 0.433% | 0.538% |
| PR-AUC | 0.506 | 0.449 | 0.464 (86x prevalence) |
| ROC-AUC | 0.997 | 0.996 | 0.995 |
| recall at FPR 1% / 0.1% | 98.1% / 24.3% | 95.5% / 19.5% | 95.0% / 16.7% |

PR-AUC looks modest next to ROC-AUC because about half of the real fraud has no chargeback label and
counts as a false positive: in the decline band precision is 50% against the labels and 99.8% against
the ground truth. That is the selection bias of label-based training, discussed in the model card
and [ADR 0008](docs/adr/0008-label-maturity-temporal-split-and-selection-bias.md). There is no baseline
model yet on purpose; gradient boosting is the usual strong baseline for tabular fraud data and an
intentional future extension ([ADR 0007](docs/adr/0007-mlp-with-embeddings-and-one-feature-pipeline.md)).

### API contracts

```bash
make contracts          # export the OpenAPI documents to contracts/openapi/
make contracts-check    # fail if the committed documents differ from the code (runs in CI)
```

The `/v1` routes of `risk-scoring-api` are implemented; those of `settlement-service` are stubs that
answer `501`. Browse `/docs` of a running service to see the full contract.

### Running the APIs without Docker

```bash
make run-risk             # http://localhost:8001 (needs Postgres: `docker compose up -d postgres`
                          # and `make migrate-risk`)
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
│   ├── shared/              # payments-shared: schemas, logging, data-contract loader and validator
│   └── datagen/             # payments-datagen: deterministic synthetic data generator (CLI)
├── services/
│   ├── risk-scoring-api/    # online fraud scoring (api, serving, training, persistence, migrations,
│   │                        #   models/current = the promoted model artifact)
│   └── settlement-service/  # batch + api around a shared domain (domain, batch, api, storage)
├── data/
│   ├── samples/             # committed output of the `sample` profile
│   └── out/                 # local runs (ignored by git)
├── contracts/
│   ├── data/                # YAML data contracts (source of truth)
│   ├── openapi/             # generated from the apps, checked in CI
│   └── examples/            # example payloads, validated by tests
├── docker/postgres/init/    # creates the risk_scoring and settlement databases
├── scripts/                 # export_openapi.py, score_sample.py, bench_risk.py
├── docs/
│   ├── architecture.md
│   ├── data-model.md        # ER diagram and notes on keys, units and the due-date cutoff
│   ├── fraud-patterns.md    # patterns, hard negatives, label delay and bias
│   ├── risk-scoring.md      # the online path, skew, idempotency, bands, failure behavior
│   ├── model-card.md        # intended use, data, metrics, limitations, ethics
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
- [ADR 0003: `payments-datagen` as a workspace package, one shared universe](docs/adr/0003-datagen-workspace-package-and-single-universe.md)
- [ADR 0004: Contract-first, YAML data contracts and generated OpenAPI](docs/adr/0004-contract-first-with-yaml-data-contracts-and-generated-openapi.md)
- [ADR 0005: Money is an integer in minor units](docs/adr/0005-money-as-integer-minor-units.md)
- [ADR 0006: v1 scoring takes precomputed behavior features](docs/adr/0006-v1-scoring-takes-precomputed-behavior-features.md)
- [ADR 0007: An MLP with embeddings, one feature pipeline, no baseline yet](docs/adr/0007-mlp-with-embeddings-and-one-feature-pipeline.md)
- [ADR 0008: Label maturity, temporal split and selection bias](docs/adr/0008-label-maturity-temporal-split-and-selection-bias.md)
- [ADR 0009: Serving concurrency, idempotent decision log, ports and adapters](docs/adr/0009-serving-concurrency-idempotent-decision-log-and-ports.md)

## Roadmap

- [x] **Phase 1: Foundation.** Workspace, shared package, both service skeletons with `/health`,
  Dockerfiles, Compose, CI, Makefile, docs.
- [x] **Phase 2: Data generators and contracts.** Data contracts, v1 API contracts with exported OpenAPI,
  a deterministic generator for one shared universe, a committed sample, validation and CI checks.
- [x] **Phase 3: risk-scoring-api.** Shared feature pipeline, label maturity and temporal splits, a
  PyTorch MLP with embeddings, evaluation report and model card, online scoring with an idempotent
  decision log (Postgres, Alembic). No baseline model yet: a gradient-boosting baseline is a planned extension.
- [ ] **Phase 4: settlement-service.** Domain rules, Polars batch pipeline, Parquet marts, read API for BI.
- [ ] **Phase 5: Polish.** Observability, performance notes, documentation pass, demo script.

## License

No license has been chosen yet.
