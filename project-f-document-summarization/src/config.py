"""Environment-backed settings."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8020
    api_key: str | None = None

    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"

    data_dir: Path = Path("data")
    jobs_db: Path = Path("data/jobs.db")

    # Webhook signing for outbound job callbacks (HMAC-SHA256 hex)
    webhook_secret: str = "dev-change-me"

    # Chunking for map-reduce (characters)
    chunk_size: int = 3500
    chunk_overlap: int = 400
    stuff_max_chars: int = 12000


def get_settings() -> Settings:
    return Settings()


settings = get_settings()
