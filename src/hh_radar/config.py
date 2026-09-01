"""Конфигурация приложения.

Всё, что зависит от окружения, читается здесь и больше нигде. Секреты берутся
только из переменных окружения или .env — в коде их нет и быть не должно.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Настройки, собранные из окружения и .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="",
    )

    # --- база ---
    database_url: str = Field(
        default="postgresql+psycopg://hh:hh@localhost:5433/hh_radar",
        alias="DATABASE_URL",
        description="DSN для SQLAlchemy. Драйвер psycopg (v3), синхронный.",
    )

    # --- доступ к API hh.ru ---
    # /vacancies закрыт: без токена приложения hh отвечает 403 forbidden.
    # Токен получается по client_credentials, см. hh_radar.hh.auth.
    hh_client_id: SecretStr | None = Field(default=None, alias="HH_CLIENT_ID")
    hh_client_secret: SecretStr | None = Field(default=None, alias="HH_CLIENT_SECRET")
    hh_access_token: SecretStr | None = Field(
        default=None,
        alias="HH_ACCESS_TOKEN",
        description="Готовый токен. Если задан — client_credentials не используется.",
    )
    hh_user_agent: str = Field(
        default="hh-radar/0.4 (github.com/Denmurzik/hh-radar)",
        alias="HH_USER_AGENT",
        description="hh требует заголовок HH-User-Agent с именем приложения и контактом.",
    )
    hh_api_base: str = Field(default="https://api.hh.ru", alias="HH_API_BASE")
    hh_token_cache: Path = Field(
        default=PROJECT_ROOT / ".hh-token-cache.json",
        alias="HH_TOKEN_CACHE",
        description="Куда класть выданный токен, чтобы не дёргать /token на каждый запуск.",
    )

    # --- вежливость к чужому API ---
    hh_requests_per_second: float = Field(default=4.0, alias="HH_RPS", gt=0)
    hh_max_retries: int = Field(default=5, alias="HH_MAX_RETRIES", ge=0)
    hh_timeout_seconds: float = Field(default=20.0, alias="HH_TIMEOUT", gt=0)

    # --- эмбеддинги ---
    # Выбор модели объяснён в README: из многоязычных, доступных в fastembed,
    # это самая лёгкая (220 МБ, CPU, без GPU). multilingual-e5-large сильнее,
    # но весит 2.24 ГБ — для репозитория, который должен запускаться у любого
    # проверяющего одной командой, это неоправданно.
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        alias="EMBEDDING_MODEL",
    )
    embedding_dim: int = Field(default=384, alias="EMBEDDING_DIM")
    embedding_backend: str = Field(
        default="fastembed",
        alias="EMBEDDING_BACKEND",
        description="fastembed | hash. hash — детерминированная заглушка для тестов и CI.",
    )
    embedding_cache_dir: Path = Field(
        default=PROJECT_ROOT / ".fastembed_cache", alias="EMBEDDING_CACHE_DIR"
    )
    chunk_chars: int = Field(default=900, alias="CHUNK_CHARS", gt=0)
    chunk_overlap_chars: int = Field(default=150, alias="CHUNK_OVERLAP_CHARS", ge=0)

    # --- профиль кандидата ---
    profile_path: Path = Field(default=PROJECT_ROOT / "profile.yaml", alias="PROFILE_PATH")

    @field_validator("embedding_backend")
    @classmethod
    def _known_backend(cls, v: str) -> str:
        allowed = {"fastembed", "hash"}
        if v not in allowed:
            raise ValueError(
                f"embedding_backend должен быть одним из {sorted(allowed)}, получено {v!r}"
            )
        return v

    @property
    def has_hh_credentials(self) -> bool:
        return self.hh_access_token is not None or (
            self.hh_client_id is not None and self.hh_client_secret is not None
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Настройки-синглтон. Кэш сбрасывается в тестах через get_settings.cache_clear()."""
    return Settings()
