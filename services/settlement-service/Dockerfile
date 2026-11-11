# Build context is the REPO ROOT (the uv workspace and uv.lock are required):
#   docker build -f services/settlement-service/Dockerfile -t settlement-service .
#
# One image, three execution modes (same code, different command):
#   API    (default CMD):  docker run --rm -p 8002:8000 settlement-service
#   Worker (override CMD): python -m settlement.batch worker
#   Batch  (override CMD): python -m settlement.batch run --date 2026-11-10
#
# Data: the dataset root is read from /data/input (mount it read-only); the marts are written to
# /data/marts, which belongs to the non-root user so a fresh named volume is writable.
# Migrations (`alembic upgrade head`) and the data contracts the pipeline validates against are inside.

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
COPY packages ./packages
COPY services ./services
# --no-editable installs the workspace members into the venv, so the runtime image needs no sources.
RUN uv sync --frozen --no-dev --no-editable --package settlement-service

FROM python:3.12-slim AS runtime
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 \
    CONTRACTS_DIR=/app/contracts \
    SETTLEMENT_INPUT_DIR=/data/input \
    SETTLEMENT_MARTS_DIR=/data/marts
RUN useradd --system --uid 10001 --no-create-home app \
    && mkdir -p /data/input /data/marts \
    && chown -R app /data/marts
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY contracts/data ./contracts/data
COPY services/settlement-service/migrations ./migrations
COPY services/settlement-service/alembic.ini ./alembic.ini
USER app
EXPOSE 8000
# /ready (not /health): healthy only when the database answers and the marts are readable.
HEALTHCHECK --interval=15s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/ready', timeout=2)"
CMD ["uvicorn", "settlement.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
