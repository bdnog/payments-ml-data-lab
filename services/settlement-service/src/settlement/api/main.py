"""FastAPI application for the settlement service (API execution mode)."""

import logging
from collections.abc import AsyncIterator, Callable
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
from settlement.api.errors import install_error_handlers
from settlement.api.routes import router
from settlement.api.schemas import ReadyResponse
from settlement.config import Settings
from settlement.persistence.repository import RunRepository, SqlRunRepository
from settlement.persistence.session import check_connection, create_db_engine, session_factory
from settlement.storage.marts import MartStore

SERVICE_NAME = "settlement-service"
DESCRIPTION = """\
Read-mostly API over the marts produced by the settlement batch, plus the registration of runs.

* The marts are Hive-partitioned Parquet; `/v1/settlements` and `/v1/reconciliation/breaks` page
  through them with opaque keyset cursors, and `/v1/exports/...` downloads a partition as Parquet.
* `POST /v1/runs` only *registers* a run (idempotent by `run_date`); a separate worker executes it.
* `daily_settlement_by_merchant` is the **ledger-expected payout**, not what the PSP confirmed.

All data is synthetic.
"""

log = logging.getLogger("settlement.api")


def create_app(
    settings: Settings | None = None,
    *,
    repository: RunRepository | None = None,
    database_check: Callable[[], None] | None = None,
) -> FastAPI:
    """Build the app. Tests pass an in-memory ``repository`` (and a ``database_check``)."""
    settings = settings or Settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.store = MartStore(settings.resolved_marts_dir())
        engine: Engine | None = None
        if repository is None:
            engine = create_db_engine(settings.database_url)
            app.state.repository = SqlRunRepository(session_factory(engine))
            app.state.database_check = lambda: check_connection(engine)
        else:
            app.state.repository = repository
            app.state.database_check = database_check or (lambda: None)
        try:
            yield
        finally:
            if engine is not None:
                engine.dispose()

    app = FastAPI(
        title=SERVICE_NAME, version=settings.app_version, description=DESCRIPTION, lifespan=lifespan
    )
    install_problem_handling(app)
    install_error_handlers(app)

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        """Liveness only: the process is up."""
        return HealthResponse(status="ok", service=SERVICE_NAME, version=settings.app_version)

    @app.get(
        "/ready",
        operation_id="ready",
        response_model=ReadyResponse,
        responses=problem_responses(503),
        tags=["ops"],
    )
    def ready(request: Request):
        """Readiness: `200` when the runs database answers and the marts directory is readable."""
        problems: list[str] = []
        try:
            request.app.state.database_check()
        except Exception as exc:  # noqa: BLE001 - any failure means "not ready"
            problems.append(f"database: {exc.__class__.__name__}")
        store: MartStore = request.app.state.store
        if not store.is_readable():
            problems.append(f"marts directory not readable: {store.root}")
        if problems:
            return problem_response(
                request,
                503,
                "Not ready",
                "; ".join(problems),
                type=problem_type("not-ready"),
            )
        return ReadyResponse(status="ready")

    app.include_router(router)
    return app


app = create_app()
