"""Service configuration, read from ``RISK_*`` environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from payments_shared.paths import repo_root

SERVICE_DIR = Path("services") / "risk-scoring-api"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RISK_")

    app_version: str = "0.1.0"
    log_level: str = "INFO"

    # Postgres (decision log). The default matches the docker-compose stack published on localhost.
    database_url: str = "postgresql+psycopg://payments:payments@localhost:5432/risk_scoring"

    # Directory of the model artifact bundle. Unset: the committed models/current of this repo.
    model_dir: Path | None = None

    # Override the decision thresholds stored in the artifact (set both or one; review <= decline).
    threshold_review: float | None = None
    threshold_decline: float | None = None

    # Inference is CPU-bound: one torch thread per worker, scale with uvicorn workers instead.
    torch_threads: int = 1

    def resolved_model_dir(self) -> Path:
        if self.model_dir is not None:
            return self.model_dir
        try:
            return repo_root() / SERVICE_DIR / "models" / "current"
        except FileNotFoundError:  # not running from a checkout (for example, in the image)
            return Path("models/current")
