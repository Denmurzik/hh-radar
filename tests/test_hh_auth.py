"""Тесты получения и кэширования токена приложения."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx

from hh_radar.config import Settings
from hh_radar.hh.auth import EXPIRY_MARGIN, HHAuthError, TokenBundle, TokenProvider

TOKEN_URL = "https://api.hh.ru/token"


def _token_response(expires_in: int = 1209600, token: str = "fresh-token") -> httpx.Response:
    return httpx.Response(
        200,
        json={"access_token": token, "token_type": "bearer", "expires_in": expires_in},
    )


class TestTokenBundle:
    def test_fresh_while_far_from_expiry(self) -> None:
        bundle = TokenBundle("t", datetime.now(UTC) + timedelta(days=1))
        assert bundle.is_fresh is True

    def test_stale_inside_the_safety_margin(self) -> None:
        """Токен, которому осталось меньше запаса, считаем протухшим заранее."""
        bundle = TokenBundle("t", datetime.now(UTC) + EXPIRY_MARGIN - timedelta(seconds=1))
        assert bundle.is_fresh is False

    def test_roundtrip_through_dict(self) -> None:
        original = TokenBundle("t", datetime.now(UTC) + timedelta(days=3))
        assert TokenBundle.from_dict(original.to_dict()) == original


class TestTokenProvider:
    @respx.mock
    def test_requests_token_with_client_credentials(self, isolated_settings: Settings) -> None:
        route = respx.post(TOKEN_URL).mock(return_value=_token_response())

        assert TokenProvider(isolated_settings).get_token() == "fresh-token"

        assert route.called
        sent = dict(pair.split("=") for pair in route.calls[0].request.content.decode().split("&"))
        assert sent["grant_type"] == "client_credentials"
        assert sent["client_id"] == "test-client"

    @respx.mock
    def test_sends_hh_user_agent_header(self, isolated_settings: Settings) -> None:
        route = respx.post(TOKEN_URL).mock(return_value=_token_response())
        TokenProvider(isolated_settings).get_token()
        assert route.calls[0].request.headers["HH-User-Agent"] == isolated_settings.hh_user_agent

    @respx.mock
    def test_second_call_does_not_hit_network(self, isolated_settings: Settings) -> None:
        route = respx.post(TOKEN_URL).mock(return_value=_token_response())
        provider = TokenProvider(isolated_settings)
        provider.get_token()
        provider.get_token()
        assert route.call_count == 1

    @respx.mock
    def test_token_survives_process_restart_via_disk_cache(
        self, isolated_settings: Settings
    ) -> None:
        route = respx.post(TOKEN_URL).mock(return_value=_token_response())
        TokenProvider(isolated_settings).get_token()

        # Новый провайдер — как новый запуск CLI. Сети быть не должно.
        assert TokenProvider(isolated_settings).get_token() == "fresh-token"
        assert route.call_count == 1

    @respx.mock
    def test_expired_cache_is_refreshed(self, isolated_settings: Settings) -> None:
        stale = TokenBundle("old", datetime.now(UTC) - timedelta(days=1))
        isolated_settings.hh_token_cache.write_text(json.dumps(stale.to_dict()), encoding="utf-8")
        respx.post(TOKEN_URL).mock(return_value=_token_response(token="new-token"))

        assert TokenProvider(isolated_settings).get_token() == "new-token"

    @respx.mock
    def test_corrupted_cache_does_not_crash(self, isolated_settings: Settings) -> None:
        isolated_settings.hh_token_cache.write_text("{не json", encoding="utf-8")
        respx.post(TOKEN_URL).mock(return_value=_token_response())

        assert TokenProvider(isolated_settings).get_token() == "fresh-token"

    def test_ready_token_from_env_skips_the_request(self, isolated_settings: Settings) -> None:
        settings = isolated_settings.model_copy(update={"hh_access_token": _secret("env-token")})
        # respx не активирован: любой сетевой вызов здесь провалил бы тест.
        assert TokenProvider(settings).get_token() == "env-token"

    def test_missing_credentials_explains_what_to_do(self, isolated_settings: Settings) -> None:
        settings = isolated_settings.model_copy(
            update={"hh_client_id": None, "hh_client_secret": None}
        )
        with pytest.raises(HHAuthError, match=re.escape("dev.hh.ru/admin")):
            TokenProvider(settings).get_token()

    @respx.mock
    def test_invalid_client_gives_actionable_message(self, isolated_settings: Settings) -> None:
        respx.post(TOKEN_URL).mock(
            return_value=httpx.Response(
                400, json={"error": "invalid_client", "error_description": "not found"}
            )
        )
        with pytest.raises(HHAuthError, match="invalid_client"):
            TokenProvider(isolated_settings).get_token()

    @respx.mock
    def test_invalidate_clears_memory_and_disk(self, isolated_settings: Settings) -> None:
        route = respx.post(TOKEN_URL).mock(return_value=_token_response())
        provider = TokenProvider(isolated_settings)
        provider.get_token()

        provider.invalidate()
        assert not isolated_settings.hh_token_cache.exists()

        provider.get_token()
        assert route.call_count == 2

    @respx.mock
    def test_malformed_token_response_is_reported(self, isolated_settings: Settings) -> None:
        respx.post(TOKEN_URL).mock(return_value=httpx.Response(200, json={"unexpected": True}))
        with pytest.raises(HHAuthError, match="неожиданный ответ"):
            TokenProvider(isolated_settings).get_token()


def _secret(value: str):  # type: ignore[no-untyped-def]
    from pydantic import SecretStr

    return SecretStr(value)
