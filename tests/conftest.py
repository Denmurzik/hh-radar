"""Общие фикстуры.

Главное здесь — ``isolated_settings``: тесты не должны видеть .env
разработчика и не должны писать кэш токена в рабочую копию.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from hh_radar.config import Settings, get_settings

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def search_page() -> dict[str, Any]:
    return load_fixture("vacancy_search_page.json")


@pytest.fixture
def vacancy_detail() -> dict[str, Any]:
    return load_fixture("vacancy_detail.json")


@pytest.fixture
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """Настройки, не зависящие от окружения машины, на которой идут тесты."""
    for name in (
        "HH_CLIENT_ID",
        "HH_CLIENT_SECRET",
        "HH_ACCESS_TOKEN",
        "DATABASE_URL",
        "EMBEDDING_BACKEND",
        "PROFILE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        HH_CLIENT_ID="test-client",
        HH_CLIENT_SECRET="test-secret",
        HH_TOKEN_CACHE=tmp_path / "token.json",
        HH_RPS=1000.0,  # тесты не должны спать
        HH_MAX_RETRIES=2,
        EMBEDDING_BACKEND="hash",
        PROFILE_PATH=tmp_path / "profile.yaml",
    )
    get_settings.cache_clear()
    yield settings
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def database_url() -> str:
    """DSN для интеграционных тестов.

    Берётся из окружения — тот же, что у приложения. Если база не отвечает,
    тесты, которые её требуют, пропускаются, а не падают: разработчик без
    поднятого docker compose должен видеть зелёный прогон юнит-тестов.
    """
    import os

    return os.environ.get("DATABASE_URL", "postgresql+psycopg://hh:hh@localhost:5433/hh_radar")


@pytest.fixture
def db_session(database_url: str):  # type: ignore[no-untyped-def]
    """Сессия в транзакции, которая откатывается после теста.

    Тест видит настоящий PostgreSQL со всеми его особенностями (генерируемый
    tsvector, pgvector, приведения типов), но ничего после себя не оставляет.
    """
    import sqlalchemy
    from sqlalchemy.orm import Session

    engine = sqlalchemy.create_engine(database_url, poolclass=sqlalchemy.pool.NullPool)
    try:
        connection = engine.connect()
    except Exception as exc:  # pragma: no cover - зависит от окружения
        pytest.skip(f"PostgreSQL недоступен ({exc.__class__.__name__}); docker compose up -d db")

    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")

    # Тест должен видеть пустую базу, даже если у разработчика в ней лежат
    # собранные вакансии. Очистка происходит внутри транзакции, которую
    # откатывают в конце, поэтому настоящие данные не страдают.
    from hh_radar.db.models import Employer, Skill, Vacancy, VacancyChunk, vacancy_skills

    session.execute(vacancy_skills.delete())
    for model in (VacancyChunk, Vacancy, Skill, Employer):
        session.execute(sqlalchemy.delete(model))
    session.flush()

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
        engine.dispose()
