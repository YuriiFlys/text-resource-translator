from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Усі налаштування читаються з оточення або `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    pg_dsn: str = Field(
        default="postgresql+psycopg://trt:trt@localhost:5432/trt",
        description="SQLAlchemy DSN однієї бази для i18n_* та схеми rag",
    )

    source_locale: str = "en"

    embedding_provider: Literal["huggingface", "openai", "hashing", "fake"] = "huggingface"
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dim: int = 768

    llm_provider: Literal["auto", "anthropic", "openai", "ollama"] = "auto"
    llm_model: str = "claude-sonnet-5-5"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_seconds: float = Field(default=120, gt=0)
    translation_batch_size: int = Field(default=20, ge=1, le=100)
    translation_max_retries: int = Field(default=1, ge=0, le=5)

    raw_dir: Path = Path("./data/raw")
    sources_file: Path = Path("./sources.yaml")

    interval_minutes: int = 30
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
