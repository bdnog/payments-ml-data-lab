.DEFAULT_GOAL := help
.PHONY: help sync lint format test run-risk run-settlement-api up down build data-sample data-full data-validate contracts contracts-check train-risk eval-risk promote-model bench-risk migrate-risk test-db demo-score settle-run settle-all verify-settlement migrate-settlement demo-settlement bench-settlement

help: ## List available targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

sync: ## Install all workspace packages and dev tools
	uv sync --all-packages

lint: ## Run ruff lint and format checks
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-fix lint issues and format the code
	uv run ruff check --fix .
	uv run ruff format .

test: ## Run the test suite for all workspace members
	uv run pytest

run-risk: ## Run risk-scoring-api locally with reload (http://localhost:8001)
	uv run uvicorn risk_scoring.api.main:app --reload --port 8001

run-settlement-api: ## Run settlement-service API locally with reload (http://localhost:8002)
	uv run uvicorn settlement.api.main:app --reload --port 8002

up: ## Start postgres, redis and both APIs with Docker Compose
	docker compose up -d --build

down: ## Stop the Docker Compose stack
	docker compose down

build: ## Build both service images
	docker compose build

data-sample: ## Regenerate the committed sample in data/samples (deterministic, seed 42)
	uv run python -m payments_datagen generate --profile sample --out data/samples

data-full: ## Generate the full profile (about 500k transactions) into data/out/full
	uv run python -m payments_datagen generate --profile full --out data/out/full

data-validate: ## Validate a generated directory (DIR=data/samples by default)
	uv run python -m payments_datagen validate $(or $(DIR),data/samples)

contracts: ## Export the OpenAPI documents to contracts/openapi
	uv run python scripts/export_openapi.py

contracts-check: contracts ## Fail if committed OpenAPI documents differ from the code
	git diff --exit-code contracts/
	test -z "$$(git ls-files --others --exclude-standard contracts/)"

RISK_DIR := services/risk-scoring-api
SETTLE_DIR := services/settlement-service
DATA ?= data/out/full

train-risk: ## Train the risk model into $(RISK_DIR)/models/runs/VERSION (VERSION=mlp-2026.11.1 DATA=data/out/full)
	@test -n "$(VERSION)" || { echo "VERSION is required, e.g. make train-risk VERSION=mlp-2026.11.1"; exit 1; }
	uv run python -m risk_scoring.training train --data $(DATA) \
		--out $(RISK_DIR)/models/runs/$(VERSION) --model-version $(VERSION)

eval-risk: ## Evaluate a run or the promoted model on a dataset (MODEL=$(RISK_DIR)/models/current DATA=data/out/full)
	uv run python -m risk_scoring.training evaluate \
		--model-dir $(or $(MODEL),$(RISK_DIR)/models/current) --data $(DATA)

promote-model: ## Copy a trained run into $(RISK_DIR)/models/current (VERSION=mlp-2026.11.1), which is committed
	@test -d "$(RISK_DIR)/models/runs/$(VERSION)" || { echo "no run $(RISK_DIR)/models/runs/$(VERSION): train it first"; exit 1; }
	rm -rf $(RISK_DIR)/models/current
	cp -R $(RISK_DIR)/models/runs/$(VERSION) $(RISK_DIR)/models/current
	@echo "promoted $(VERSION): review and commit $(RISK_DIR)/models/current"

bench-risk: ## In-process latency of 1,000 single-request scorings with the promoted model (p50, p95)
	uv run python scripts/bench_risk.py

migrate-risk: ## Apply the Alembic migrations to the risk_scoring database (needs Postgres: make up)
	cd $(RISK_DIR) && uv run alembic upgrade head

test-db: ## Start Postgres and run the SQL tests (risk decision log, settlement runs queue) on throwaway databases
	docker compose up -d --wait postgres
	docker compose exec -T postgres psql -U payments -d payments -c "CREATE DATABASE risk_scoring_test" 2>/dev/null || true
	docker compose exec -T postgres psql -U payments -d payments -c "CREATE DATABASE settlement_test" 2>/dev/null || true
	RISK_TEST_DATABASE_URL=postgresql+psycopg://payments:payments@localhost:$(or $(POSTGRES_PORT),5432)/risk_scoring_test \
	SETTLEMENT_TEST_DATABASE_URL=postgresql+psycopg://payments:payments@localhost:$(or $(POSTGRES_PORT),5432)/settlement_test \
		uv run pytest $(RISK_DIR)/tests/persistence $(SETTLE_DIR)/tests/persistence -v

demo-score: ## Replay sample rows through the running API (make up first): scores, batch, replay, 409
	uv run python scripts/score_sample.py --url http://localhost:$(or $(RISK_API_PORT),8001)

# The settlement CLI takes absolute paths (the marts default to data/marts, ignored by git).
SETTLE_DATA = $(abspath $(or $(DATA),data/samples))
SETTLE_MARTS = $(abspath $(or $(MARTS),data/marts))

settle-run: ## Settle one date, bypassing the queue (DATE=2026-01-15 DATA=data/samples MARTS=data/marts)
	@test -n "$(DATE)" || { echo "DATE is required, e.g. make settle-run DATE=2026-01-15"; exit 1; }
	uv run python -m settlement.batch run --date $(DATE) --input-dir $(SETTLE_DATA) --marts-dir $(SETTLE_MARTS)

settle-all: ## Settle every date the PSP file covers, in one pass (DATA=data/out/full)
	uv run python -m settlement.batch run-range --all --input-dir $(SETTLE_DATA) --marts-dir $(SETTLE_MARTS)

verify-settlement: ## Compare the breaks mart with the generator's answer key: 100% or non-zero exit
	uv run python -m settlement.batch verify --input-dir $(SETTLE_DATA) --marts-dir $(SETTLE_MARTS)

migrate-settlement: ## Apply the Alembic migrations to the settlement database (needs Postgres: make up)
	cd $(SETTLE_DIR) && uv run alembic upgrade head

demo-settlement: ## Drive the running stack: POST a run, wait for the worker, page, export, replay
	uv run python scripts/demo_settlement.py --url http://localhost:$(or $(SETTLEMENT_API_PORT),8002) $(if $(DATE),--date $(DATE))

bench-settlement: ## Time one date and the whole range on a dataset, and the marts size (DATA=data/out/full)
	uv run python scripts/bench_settlement.py --data $(or $(DATA),data/out/full)
