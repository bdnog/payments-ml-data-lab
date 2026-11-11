"""FastAPI application for the settlement service (API execution mode)."""

from fastapi import FastAPI

from payments_shared import HealthResponse, configure_logging
from payments_shared.problem import install_problem_handling
from settlement.api.routes import router
from settlement.config import Settings

SERVICE_NAME = "settlement-service"
DESCRIPTION = """\
Read-mostly API over the marts produced by the settlement batch. **v1 contract**: only `/health`
is implemented; the `/v1` routes answer `501`.
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
