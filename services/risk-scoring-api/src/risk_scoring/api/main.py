"""FastAPI application for the risk-scoring service."""

from fastapi import FastAPI

from payments_shared import HealthResponse, configure_logging
from payments_shared.problem import install_problem_handling
from risk_scoring.api.routes import router
from risk_scoring.config import Settings

SERVICE_NAME = "risk-scoring-api"
DESCRIPTION = """\
Online fraud scoring. **v1 contract**: the caller sends the transaction together with its
precomputed `behavior` features. Only `/health` is implemented; the `/v1` routes answer `501`.
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)

    app = FastAPI(title=SERVICE_NAME, version=settings.app_version, description=DESCRIPTION)
    install_problem_handling(app)

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", service=SERVICE_NAME, version=settings.app_version)

    app.include_router(router)
    return app


app = create_app()
