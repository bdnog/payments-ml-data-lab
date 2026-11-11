.DEFAULT_GOAL := help
.PHONY: help sync lint format test run-risk run-settlement-api up down build data-sample

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
