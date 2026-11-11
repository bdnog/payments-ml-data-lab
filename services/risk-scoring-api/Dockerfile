# Build context is the REPO ROOT (the uv workspace and uv.lock are required):
#   docker build -f services/risk-scoring-api/Dockerfile -t risk-scoring-api .
#   docker run --rm -p 8001:8000 -e RISK_DATABASE_URL=postgresql+psycopg://... risk-scoring-api
#
# The image contains the promoted model (models/current), the data contract the feature pipeline
# reads (contracts/data), and the Alembic migrations (run them with `alembic upgrade head`).
# Scale with uvicorn workers: -e WEB_CONCURRENCY=4 (uvicorn reads it; one torch thread per worker).

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
COPY packages ./packages
COPY services ./services
# --no-editable installs the workspace members into the venv, so the runtime image needs no sources.
# On Linux uv resolves torch from the CPU-only index (see the root pyproject), not the CUDA wheels.
RUN uv sync --frozen --no-dev --no-editable --package risk-scoring-api

FROM python:3.12-slim AS runtime
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 \
    CONTRACTS_DIR=/app/contracts \
    RISK_MODEL_DIR=/app/models/current
RUN useradd --system --uid 10001 --no-create-home app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY contracts/data ./contracts/data
COPY services/risk-scoring-api/models/current ./models/current
COPY services/risk-scoring-api/migrations ./migrations
COPY services/risk-scoring-api/alembic.ini ./alembic.ini
USER app
EXPOSE 8000
# /ready (not /health): the container is only healthy once the model is loaded and verified.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/ready', timeout=2)"
CMD ["uvicorn", "risk_scoring.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
