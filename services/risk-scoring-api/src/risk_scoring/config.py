"""Service configuration, read from ``RISK_*`` environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RISK_")

    app_version: str = "0.1.0"
    log_level: str = "INFO"
