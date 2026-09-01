"""Клиент API hh.ru.

Здесь живут три вещи, которых нет в примере из документации и без которых
сборщик не доживает до конца первой тысячи вакансий:

1. **Обход потолка выдачи.** hh не отдаёт больше ``MAX_RESULTS`` элементов
   на один поисковый запрос, сколько бы страниц ни просить. Если по запросу
   найдено больше — интервал дат делится пополам и каждая половина
   выкачивается отдельно (:meth:`HHClient.iter_vacancies`). Рекурсия
   останавливается, когда окно сжалось до часа: если и в один час больше
   двух тысяч вакансий, честнее вернуть что есть и сказать об этом в логе,
   чем молча потерять данные.

2. **Ретраи, различающие временное и постоянное.** 429 и 5xx — ждём и
   повторяем с экспоненциальной паузой, уважая ``Retry-After``. 403
   ``bad_authorization`` — токен протух раньше срока, сбрасываем его и
   пробуем ещё раз. 404 — вакансию удалили, это не ошибка сборщика.

3. **Ограничение частоты.** Чужой публичный API — не полигон: между
   запросами выдерживается минимальный интервал (``HH_RPS``).
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import httpx

from hh_radar.config import Settings, get_settings
from hh_radar.hh.auth import TokenProvider

logger = logging.getLogger(__name__)

#: Жёсткий потолок выдачи hh на один поисковый запрос.
MAX_RESULTS = 2000
#: Максимальный размер страницы, который принимает /vacancies.
MAX_PER_PAGE = 100
#: Ниже этого окна дробить дальше бессмысленно.
MIN_WINDOW = timedelta(hours=1)


class HHError(RuntimeError):
    """Ошибка обращения к API hh, которую не удалось починить ретраем."""


class HHNotFoundError(HHError):
    """Запрошенного объекта нет — вакансия снята или скрыта."""


@dataclass(frozen=True, slots=True)
class SearchPage:
    """Одна страница поисковой выдачи."""

    items: list[dict[str, Any]]
    page: int
    pages: int
    found: int
    per_page: int


class RateLimiter:
    """Простейший ограничитель: не чаще N запросов в секунду.

    Не токен-бакет намеренно — равномерный интервал вежливее к чужому API,
    чем всплеск на весь бакет с последующей паузой.
    """

    def __init__(self, requests_per_second: float) -> None:
        self._min_interval = 1.0 / requests_per_second
        # -inf, а не 0: иначе первый же запрос за время жизни процесса
        # ждал бы целый интервал впустую.
        self._last_call = float("-inf")

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()


class HHClient:
    """Синхронный клиент API hh.ru.

    Используется как контекстный менеджер::

        with HHClient() as client:
            for vacancy in client.iter_vacancies(text="MCP", date_from=..., date_to=...):
                ...
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        http_client: httpx.Client | None = None,
        token_provider: TokenProvider | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(
            base_url=self._settings.hh_api_base,
            timeout=self._settings.hh_timeout_seconds,
            headers={"HH-User-Agent": self._settings.hh_user_agent},
        )
        self._tokens = token_provider or TokenProvider(self._settings)
        self._limiter = RateLimiter(self._settings.hh_requests_per_second)

    def __enter__(self) -> HHClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    # ------------------------------------------------------------ endpoints --

    def search(
        self,
        *,
        text: str | None = None,
        page: int = 0,
        per_page: int = MAX_PER_PAGE,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        area: int | str | None = None,
        **extra: Any,
    ) -> SearchPage:
        """Одна страница поиска вакансий."""
        params: dict[str, Any] = {
            "page": page,
            "per_page": min(per_page, MAX_PER_PAGE),
            **{k: v for k, v in extra.items() if v is not None},
        }
        if text:
            params["text"] = text
        if area is not None:
            params["area"] = area
        if date_from is not None:
            params["date_from"] = _isoformat(date_from)
        if date_to is not None:
            params["date_to"] = _isoformat(date_to)

        payload = self._request("GET", "/vacancies", params=params)
        return SearchPage(
            items=payload.get("items", []),
            page=payload.get("page", page),
            pages=payload.get("pages", 0),
            found=payload.get("found", 0),
            per_page=payload.get("per_page", per_page),
        )

    def get_vacancy(self, vacancy_id: int | str) -> dict[str, Any]:
        """Полная карточка вакансии.

        Нужна отдельным запросом: поиск отдаёт только ``snippet`` с обрезанными
        обрывками текста, а нам нужны ``description`` целиком и ``key_skills``.
        """
        return self._request("GET", f"/vacancies/{vacancy_id}")

    def get_areas(self) -> list[dict[str, Any]]:
        """Справочник регионов. Открыт без авторизации, но ходим единообразно."""
        payload = self._request("GET", "/areas")
        return payload if isinstance(payload, list) else []

    # -------------------------------------------------------------- crawling --

    def iter_vacancies(
        self,
        *,
        text: str | None = None,
        date_from: datetime,
        date_to: datetime,
        area: int | str | None = None,
        **extra: Any,
    ) -> Iterator[dict[str, Any]]:
        """Все вакансии за интервал, с обходом потолка в 2000 элементов.

        Алгоритм: спросить, сколько всего найдено за окно. Если влезает в
        потолок — просто пролистать страницы. Если нет — разрезать окно
        пополам и повторить для каждой половины.
        """
        probe = self.search(
            text=text, page=0, per_page=1, date_from=date_from, date_to=date_to, area=area, **extra
        )
        if probe.found == 0:
            return

        if probe.found > MAX_RESULTS:
            window = date_to - date_from
            if window <= MIN_WINDOW:
                logger.warning(
                    "за окно %s..%s найдено %d вакансий при потолке %d; "
                    "дробить дальше некуда, часть данных будет пропущена",
                    _isoformat(date_from),
                    _isoformat(date_to),
                    probe.found,
                    MAX_RESULTS,
                )
            else:
                middle = date_from + window / 2
                logger.info(
                    "окно %s..%s даёт %d > %d, делим пополам",
                    _isoformat(date_from),
                    _isoformat(date_to),
                    probe.found,
                    MAX_RESULTS,
                )
                yield from self.iter_vacancies(
                    text=text, date_from=date_from, date_to=middle, area=area, **extra
                )
                yield from self.iter_vacancies(
                    text=text, date_from=middle, date_to=date_to, area=area, **extra
                )
                return

        page = 0
        while True:
            chunk = self.search(
                text=text,
                page=page,
                per_page=MAX_PER_PAGE,
                date_from=date_from,
                date_to=date_to,
                area=area,
                **extra,
            )
            yield from chunk.items
            page += 1
            if page >= chunk.pages or page * MAX_PER_PAGE >= MAX_RESULTS:
                break

    # ------------------------------------------------------------- transport --

    def _request(self, method: str, path: str, *, params: dict[str, Any] | None = None) -> Any:
        settings = self._settings
        last_error: Exception | None = None

        for attempt in range(settings.hh_max_retries + 1):
            self._limiter.wait()
            headers = {"Authorization": f"Bearer {self._tokens.get_token()}"}
            try:
                response = self._http.request(method, path, params=params, headers=headers)
            except httpx.TransportError as exc:
                last_error = exc
                self._sleep_before_retry(attempt, reason=f"сетевая ошибка: {exc}")
                continue

            if response.status_code == httpx.codes.OK:
                return response.json()

            if response.status_code == httpx.codes.NOT_FOUND:
                raise HHNotFoundError(f"hh: объект не найден ({method} {path})")

            if response.status_code == httpx.codes.FORBIDDEN and _is_bad_auth(response):
                # Токен отозвали или он протух раньше заявленного срока.
                # Один раз имеет смысл сходить за новым, дальше — сдаёмся.
                logger.info("hh отверг токен, запрашиваем новый")
                self._tokens.invalidate()
                last_error = HHError("hh: bad_authorization")
                if attempt == 0:
                    continue
                raise HHError(
                    "hh отклоняет токен приложения. Проверьте HH_CLIENT_ID/HH_CLIENT_SECRET "
                    "и что приложение на dev.hh.ru активно."
                )

            if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
                self._sleep_before_retry(attempt, reason="429", retry_after=_retry_after(response))
                last_error = HHError("hh: 429 Too Many Requests")
                continue

            if response.status_code >= httpx.codes.INTERNAL_SERVER_ERROR:
                last_error = HHError(f"hh: {response.status_code}")
                self._sleep_before_retry(attempt, reason=str(response.status_code))
                continue

            raise HHError(
                f"hh вернул {response.status_code} на {method} {path}: {response.text[:300]}"
            )

        raise HHError(
            f"не удалось выполнить {method} {path} за {settings.hh_max_retries + 1} попыток"
        ) from last_error

    def _sleep_before_retry(
        self, attempt: int, *, reason: str, retry_after: float | None = None
    ) -> None:
        """Экспоненциальная пауза с джиттером.

        Джиттер нужен, чтобы несколько параллельных сборщиков не били в API
        синхронно после общей паузы.
        """
        if retry_after is not None:
            delay = retry_after
        else:
            delay = min(2.0**attempt, 30.0) * (0.5 + random.random())
        logger.warning("повтор запроса через %.1f с (%s), попытка %d", delay, reason, attempt + 1)
        time.sleep(delay)


def _isoformat(moment: datetime) -> str:
    """hh принимает даты в ISO 8601; секунды и таймзона допустимы."""
    return moment.replace(microsecond=0).isoformat()


def _is_bad_auth(response: httpx.Response) -> bool:
    try:
        body = response.json()
    except ValueError:
        return False
    errors = body.get("errors") if isinstance(body, dict) else None
    if not isinstance(errors, list):
        return False
    return any(item.get("type") == "oauth" for item in errors if isinstance(item, dict))


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
