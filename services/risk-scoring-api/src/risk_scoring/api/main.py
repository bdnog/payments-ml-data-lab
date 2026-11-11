"""FastAPI application for the risk-scoring service."""

from fastapi import FastAPI

from payments_shared import HealthResponse, configure_logging
from risk_scoring.config import Settings

SERVICE_NAME = "risk-scoring-api"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)

    app = FastAPI(title=SERVICE_NAME, version=settings.app_version)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", service=SERVICE_NAME, version=settings.app_version)

    return app


app = create_app()
