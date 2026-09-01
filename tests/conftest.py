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
