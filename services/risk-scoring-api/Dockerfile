# Build context is the REPO ROOT (the uv workspace and uv.lock are required):
#   docker build -f services/risk-scoring-api/Dockerfile -t risk-scoring-api .
#   docker run --rm -p 8001:8000 risk-scoring-api

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
COPY packages ./packages
COPY services ./services
# --no-editable installs the workspace members into the venv, so the runtime image needs no sources.
RUN uv sync --frozen --no-dev --no-editable --package risk-scoring-api

FROM python:3.12-slim AS runtime
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
RUN useradd --system --uid 10001 --no-create-home app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["uvicorn", "risk_scoring.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
