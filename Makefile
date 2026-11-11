.DEFAULT_GOAL := help
.PHONY: help sync lint format test run-risk run-settlement-api up down build data-sample data-full data-validate contracts contracts-check train-risk eval-risk promote-model

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
