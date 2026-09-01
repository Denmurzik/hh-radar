"""Тесты клиента hh: пагинация, обход потолка выдачи, ретраи.

Сеть замокана через respx. Отдельно проверяется то, ради чего клиент вообще
написан руками, а не взят из примера: поведение на 429, на протухшем токене
и на запросе, который не помещается в потолок в 2000 элементов.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
import respx

from hh_radar.config import Settings
from hh_radar.hh.auth import TokenProvider
from hh_radar.hh.client import (
    MAX_PER_PAGE,
    MAX_RESULTS,
    HHClient,
    HHError,
    HHNotFoundError,
    RateLimiter,
)

VACANCIES_URL = "https://api.hh.ru/vacancies"


@pytest.fixture
def client(isolated_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> HHClient:
    """Клиент с заранее выданным токеном и без пауз между попытками."""
    monkeypatch.setattr("hh_radar.hh.client.time.sleep", lambda _: None)
    settings = isolated_settings
    provider = TokenProvider(settings)
    monkeypatch.setattr(provider, "get_token", lambda: "test-token")
    return HHClient(settings, token_provider=provider)


def _page(items: list[dict[str, Any]], *, found: int, pages: int, page: int = 0) -> httpx.Response:
    return httpx.Response(
        200,
        json={"items": items, "found": found, "pages": pages, "page": page, "per_page": 100},
    )


def _stub_items(count: int, start: int = 0) -> list[dict[str, Any]]:
    return [{"id": str(start + i), "name": f"Вакансия {start + i}"} for i in range(count)]


class TestRateLimiter:
    def test_first_call_does_not_wait(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Пустая пауза на старте процесса — бессмысленная задержка."""
        slept: list[float] = []
        monkeypatch.setattr("hh_radar.hh.client.time.sleep", slept.append)
        monkeypatch.setattr("hh_radar.hh.client.time.monotonic", lambda: 0.0)

        RateLimiter(requests_per_second=4.0).wait()

        assert slept == []

    def test_waits_out_the_remaining_interval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        slept: list[float] = []
        monkeypatch.setattr("hh_radar.hh.client.time.sleep", slept.append)
        # wait() дёргает monotonic дважды: на замер и на запись момента.
        clock = iter([0.0, 0.0, 0.05, 0.05])
        monkeypatch.setattr("hh_radar.hh.client.time.monotonic", lambda: next(clock))

        limiter = RateLimiter(requests_per_second=4.0)  # интервал 0.25 с
        limiter.wait()  # первый вызов не ждёт, но запоминает момент 0.0
        limiter.wait()  # прошло 0.05 — досыпаем 0.2

        assert slept == [pytest.approx(0.2, abs=1e-6)]


class TestSearch:
    @respx.mock
    def test_authorization_header_is_sent(self, client: HHClient) -> None:
        route = respx.get(VACANCIES_URL).mock(return_value=_page([], found=0, pages=0))
        client.search(text="n8n")
        assert route.calls[0].request.headers["Authorization"] == "Bearer test-token"

    @respx.mock
    def test_dates_are_serialized_without_microseconds(self, client: HHClient) -> None:
        route = respx.get(VACANCIES_URL).mock(return_value=_page([], found=0, pages=0))
        moment = datetime(2026, 8, 1, 12, 30, 45, 123456, tzinfo=UTC)
        client.search(date_from=moment, date_to=moment + timedelta(days=1))

        params = route.calls[0].request.url.params
        assert "." not in params["date_from"]
        assert params["date_from"].startswith("2026-08-01T12:30:45")

    @respx.mock
    def test_per_page_is_capped(self, client: HHClient) -> None:
        route = respx.get(VACANCIES_URL).mock(return_value=_page([], found=0, pages=0))
        client.search(per_page=500)
        assert route.calls[0].request.url.params["per_page"] == str(MAX_PER_PAGE)

    @respx.mock
    def test_empty_text_is_not_sent_as_param(self, client: HHClient) -> None:
        route = respx.get(VACANCIES_URL).mock(return_value=_page([], found=0, pages=0))
        client.search(text="")
        assert "text" not in route.calls[0].request.url.params


class TestRetries:
    @respx.mock
    def test_429_is_retried_and_then_succeeds(self, client: HHClient) -> None:
        respx.get(VACANCIES_URL).mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "1"}, json={}),
                _page(_stub_items(1), found=1, pages=1),
            ]
        )
        assert client.search().found == 1

    @respx.mock
    def test_server_error_is_retried(self, client: HHClient) -> None:
        respx.get(VACANCIES_URL).mock(
            side_effect=[httpx.Response(503, json={}), _page(_stub_items(2), found=2, pages=1)]
        )
        assert len(client.search().items) == 2

    @respx.mock
    def test_gives_up_after_max_retries(self, client: HHClient) -> None:
        respx.get(VACANCIES_URL).mock(return_value=httpx.Response(500, json={}))
        with pytest.raises(HHError, match="попыток"):
            client.search()

    @respx.mock
    def test_transport_error_is_retried(self, client: HHClient) -> None:
        respx.get(VACANCIES_URL).mock(
            side_effect=[httpx.ConnectError("сеть отвалилась"), _page([], found=0, pages=0)]
        )
        assert client.search().found == 0

    @respx.mock
    def test_404_is_a_distinct_error(self, client: HHClient) -> None:
        respx.get("https://api.hh.ru/vacancies/1").mock(return_value=httpx.Response(404, json={}))
        with pytest.raises(HHNotFoundError):
            client.get_vacancy(1)

    @respx.mock
    def test_revoked_token_triggers_refresh_and_one_retry(
        self, isolated_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("hh_radar.hh.client.time.sleep", lambda _: None)
        provider = TokenProvider(isolated_settings)
        tokens = iter(["stale", "fresh", "fresh"])
        invalidated: list[bool] = []
        monkeypatch.setattr(provider, "get_token", lambda: next(tokens))
        monkeypatch.setattr(provider, "invalidate", lambda: invalidated.append(True))

        respx.get(VACANCIES_URL).mock(
            side_effect=[
                httpx.Response(
                    403, json={"errors": [{"value": "bad_authorization", "type": "oauth"}]}
                ),
                _page(_stub_items(1), found=1, pages=1),
            ]
        )

        with HHClient(isolated_settings, token_provider=provider) as client:
            assert client.search().found == 1
        assert invalidated == [True]

    @respx.mock
    def test_forbidden_without_oauth_marker_is_not_retried(self, client: HHClient) -> None:
        """403 не по поводу токена — это отказ, а не повод сбрасывать авторизацию."""
        route = respx.get(VACANCIES_URL).mock(
            return_value=httpx.Response(403, json={"errors": [{"type": "forbidden"}]})
        )
        with pytest.raises(HHError):
            client.search()
        assert route.call_count == 1


class TestIterVacancies:
    @respx.mock
    def test_walks_all_pages(self, client: HHClient) -> None:
        respx.get(VACANCIES_URL).mock(
            side_effect=[
                _page(_stub_items(1), found=150, pages=2),  # разведочный запрос
                _page(_stub_items(100, start=0), found=150, pages=2, page=0),
                _page(_stub_items(50, start=100), found=150, pages=2, page=1),
            ]
        )
        window = _window()
        collected = list(client.iter_vacancies(**window))
        assert len(collected) == 150

    @respx.mock
    def test_empty_window_makes_one_request(self, client: HHClient) -> None:
        route = respx.get(VACANCIES_URL).mock(return_value=_page([], found=0, pages=0))
        assert list(client.iter_vacancies(**_window())) == []
        assert route.call_count == 1

    @respx.mock
    def test_splits_window_when_over_the_ceiling(self, client: HHClient) -> None:
        """Больше потолка за окно — окно режется пополам, обе половины качаются."""
        responses = [
            _page(_stub_items(1), found=MAX_RESULTS + 1, pages=100),  # разведка по всему окну
            _page(_stub_items(1), found=1, pages=1),  # разведка по первой половине
            _page(_stub_items(1, start=10), found=1, pages=1),
            _page(_stub_items(1), found=1, pages=1),  # разведка по второй половине
            _page(_stub_items(1, start=20), found=1, pages=1),
        ]
        respx.get(VACANCIES_URL).mock(side_effect=responses)

        collected = list(client.iter_vacancies(**_window(delta=timedelta(days=8))))
        assert [item["id"] for item in collected] == ["10", "20"]

    @respx.mock
    def test_stops_splitting_at_the_minimum_window(self, client: HHClient) -> None:
        """Час с перебором вакансий — берём что дают и пишем предупреждение."""
        respx.get(VACANCIES_URL).mock(
            side_effect=[
                _page(_stub_items(1), found=MAX_RESULTS + 1, pages=100),
                _page(_stub_items(100), found=MAX_RESULTS + 1, pages=100, page=0),
                *[
                    _page(_stub_items(100, start=i * 100), found=MAX_RESULTS + 1, pages=100, page=i)
                    for i in range(1, 20)
                ],
            ]
        )
        collected = list(client.iter_vacancies(**_window(delta=timedelta(hours=1))))
        assert len(collected) == MAX_RESULTS


def _window(*, delta: timedelta = timedelta(days=1)) -> dict[str, datetime]:
    end = datetime(2026, 8, 30, tzinfo=UTC)
    return {"date_from": end - delta, "date_to": end}
