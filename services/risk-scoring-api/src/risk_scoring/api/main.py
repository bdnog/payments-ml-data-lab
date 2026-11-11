"""FastAPI application for the risk-scoring service."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from sqlalchemy import Engine

from payments_shared import HealthResponse, configure_logging
from payments_shared.problem import (
    install_problem_handling,
    problem_response,
    problem_responses,
    problem_type,
)
from risk_scoring.api.errors import install_error_handlers
from risk_scoring.api.routes import router
from risk_scoring.api.schemas import ReadyResponse
from risk_scoring.config import Settings
from risk_scoring.persistence.session import (
    RepositoryFactory,
    create_db_engine,
    sql_repository_factory,
)
from risk_scoring.serving.scorer import Scorer

SERVICE_NAME = "risk-scoring-api"
DESCRIPTION = """\
Online fraud scoring with a small neural network (PyTorch MLP with embeddings), trained on
**synthetic** data: the metrics say nothing about real fraud. The caller sends the transaction with
its precomputed `behavior` features (v1 contract, ADR 0006); every decision is logged idempotently.

`score` is a ranking score, not a calibrated probability. Reason codes are heuristic explanations.
"""

log = logging.getLogger("risk_scoring.api")


def create_app(
    settings: Settings | None = None, *, repository_factory: RepositoryFactory | None = None
) -> FastAPI:
    """Build the app. ``repository_factory`` replaces Postgres (tests pass an in-memory one)."""
    settings = settings or Settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Load and verify the artifact once. A failure does not stop the app: it starts, /ready
        # answers 503 and every scoring route answers 503, so an orchestrator can see why.
        app.state.scorer = None
        app.state.load_error = None
        model_dir = settings.resolved_model_dir()
        try:
            app.state.scorer = Scorer.load(model_dir, settings)
            log.info("model loaded version=%s dir=%s", app.state.scorer.model_version, model_dir)
        except Exception as exc:  # noqa: BLE001 - any failure means "not ready", whatever it is
            app.state.load_error = f"{exc.__class__.__name__}: {exc}"
            log.error("model failed to load dir=%s error=%s", model_dir, app.state.load_error)

        engine: Engine | None = None
        if repository_factory is None:
            engine = create_db_engine(settings.database_url)
            app.state.repository_factory = sql_repository_factory(engine)
        else:
            app.state.repository_factory = repository_factory
        try:
            yield
        finally:
            if engine is not None:
                engine.dispose()

    app = FastAPI(
        title=SERVICE_NAME,
        version=settings.app_version,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    install_problem_handling(app)
    install_error_handlers(app)

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        """Liveness only: the process is up. It says nothing about the model or the database."""
        return HealthResponse(status="ok", service=SERVICE_NAME, version=settings.app_version)

    @app.get(
        "/ready",
        operation_id="ready",
        response_model=ReadyResponse,
        responses=problem_responses(503),
        tags=["ops"],
    )
    def ready(request: Request):
        """Readiness: `200` when the model and its thresholds are loaded, else `503`."""
        scorer: Scorer | None = getattr(request.app.state, "scorer", None)
        if scorer is None:
            error = getattr(request.app.state, "load_error", None) or "the model is not loaded"
            return problem_response(
                request,
                503,
                "Not ready",
                f"The service cannot score yet: {error}",
                type=problem_type("not-ready"),
            )
        return ReadyResponse(status="ready", model_version=scorer.model_version)

    app.include_router(router)
    return app


app = create_app()
