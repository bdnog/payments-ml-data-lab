"""Service configuration, read from ``SETTLEMENT_*`` environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from payments_shared.paths import repo_root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SETTLEMENT_")

    app_version: str = "0.1.0"
    log_level: str = "INFO"

    # Postgres (the `runs` table). The default matches the Compose stack published on localhost.
    database_url: str = "postgresql+psycopg://payments:payments@localhost:5432/settlement"

    # Where the pipeline reads the Phase 2 dataset root (reference/ and settlement/) and where it
    # writes the marts. Unset: data/samples and data/marts of the repository checkout.
    input_dir: Path | None = None
    marts_dir: Path | None = None

    # Worker: how long to sleep when the queue is empty, when a RUNNING run counts as abandoned,
    # and how many times a run may be attempted before it is marked FAILED.
    worker_poll_seconds: float = 2.0
    stale_after_seconds: int = 900
    max_attempts: int = 3

    def resolved_input_dir(self) -> Path:
        return self.input_dir or _checkout_dir("data", "samples")

    def resolved_marts_dir(self) -> Path:
        return self.marts_dir or _checkout_dir("data", "marts")


def _checkout_dir(*parts: str) -> Path:
    try:
        return repo_root().joinpath(*parts)
    except FileNotFoundError as exc:  # not a checkout (for example, in the image)
        raise RuntimeError(
            "set SETTLEMENT_INPUT_DIR and SETTLEMENT_MARTS_DIR: no repository checkout found"
        ) from exc
