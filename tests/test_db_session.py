"""Тесты подключения к базе.

Главное здесь — таймаут. MCP-сервер общается с Claude Desktop по stdio, и
запрос без ответа выглядит для клиента не как ошибка, а как зависший сервер.
Без явного connect_timeout psycopg ждёт столько, сколько разрешит операционная
система: замеренное поведение до этой правки — больше пяти минут.
"""

from __future__ import annotations

import time

import pytest

from hh_radar.db.session import CONNECT_TIMEOUT_SECONDS, connect_args, ping, reset_caches


@pytest.fixture(autouse=True)
def _clean_caches():  # type: ignore[no-untyped-def]
    reset_caches()
    yield
    reset_caches()


class TestEngineConfiguration:
    def test_connect_timeout_is_passed_to_the_driver(self) -> None:
        assert connect_args()["connect_timeout"] == CONNECT_TIMEOUT_SECONDS

    def test_timeout_is_short_enough_to_notice(self) -> None:
        """Больше десяти секунд клиент воспримет как зависание, а не как ошибку."""
        assert 1 <= CONNECT_TIMEOUT_SECONDS <= 10


@pytest.mark.integration
class TestPing:
    def test_unreachable_host_fails_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 10.255.255.1 — адрес из зарезервированного диапазона, который
        # не отвечает и не отказывает: пакеты просто уходят в пустоту.
        # Именно этот случай раньше приводил к бесконечному ожиданию.
        monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@10.255.255.1:5433/db")
        from hh_radar.config import get_settings

        get_settings.cache_clear()
        reset_caches()

        started = time.monotonic()
        assert ping() is False
        elapsed = time.monotonic() - started

        assert elapsed < CONNECT_TIMEOUT_SECONDS * 3, f"ping занял {elapsed:.1f} с"
        get_settings.cache_clear()
