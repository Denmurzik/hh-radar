"""Авторизация в API hh.ru.

Историческая справка, из-за которой этот модуль вообще существует: раньше поиск
вакансий отвечал на анонимный GET. Сейчас нет — ``GET /vacancies`` без токена
возвращает 403 ``{"errors": [{"value": "bad_authorization", "type": "oauth"}]}``,
хотя справочники ``/dictionaries`` и ``/areas`` по-прежнему открыты.

Нужен токен приложения. Он выдаётся по ``client_credentials``: приложение
регистрируется на https://dev.hh.ru/admin, оттуда берутся client_id и
client_secret, и они меняются на access_token без участия пользователя.
Refresh-токена у этого гранта нет — истёкший токен просто запрашивается заново.

Токен живёт долго (порядок недель), поэтому дёргать ``/token`` на каждый запуск
CLI незачем: он кладётся в файл из ``settings.hh_token_cache``, который лежит
в .gitignore.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from hh_radar.config import Settings, get_settings

logger = logging.getLogger(__name__)

#: За сколько до формального истечения считать токен протухшим.
#: Защищает от гонки «токен был жив на проверке и умер в полёте».
EXPIRY_MARGIN = timedelta(minutes=5)

TOKEN_URL_PATH = "/token"


class HHAuthError(RuntimeError):
    """Не удалось получить токен. Текст сообщения рассчитан на человека."""


@dataclass(frozen=True, slots=True)
class TokenBundle:
    """Выданный токен и момент, после которого он считается непригодным."""

    access_token: str
    expires_at: datetime

    @property
    def is_fresh(self) -> bool:
        return datetime.now(UTC) < self.expires_at - EXPIRY_MARGIN

    def to_dict(self) -> dict[str, str]:
        return {
            "access_token": self.access_token,
            "expires_at": self.expires_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, str]) -> TokenBundle:
        return cls(
            access_token=raw["access_token"],
            expires_at=datetime.fromisoformat(raw["expires_at"]),
        )


class TokenProvider:
    """Отдаёт валидный access_token, обновляя его по необходимости.

    Три источника, в порядке приоритета:

    1. ``HH_ACCESS_TOKEN`` в окружении — готовый токен, ничего запрашивать не надо.
    2. Кэш на диске, если он ещё не протух.
    3. ``POST /token`` с client_credentials.
    """

    def __init__(
        self, settings: Settings | None = None, client: httpx.Client | None = None
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._cached: TokenBundle | None = None

    # ------------------------------------------------------------------ api --

    def get_token(self) -> str:
        settings = self._settings

        if settings.hh_access_token is not None:
            return settings.hh_access_token.get_secret_value()

        if self._cached is not None and self._cached.is_fresh:
            return self._cached.access_token

        from_disk = self._read_cache(settings.hh_token_cache)
        if from_disk is not None and from_disk.is_fresh:
            self._cached = from_disk
            return from_disk.access_token

        bundle = self._request_token()
        self._cached = bundle
        self._write_cache(settings.hh_token_cache, bundle)
        return bundle.access_token

    def invalidate(self) -> None:
        """Забыть текущий токен.

        Вызывается клиентом, когда hh ответил 403 ``bad_authorization``: токен
        мог быть отозван раньше срока, и следующий запрос должен идти уже
        со свежим.
        """
        self._cached = None
        cache_path = self._settings.hh_token_cache
        cache_path.unlink(missing_ok=True)

    # -------------------------------------------------------------- internals --

    def _request_token(self) -> TokenBundle:
        settings = self._settings
        if settings.hh_client_id is None or settings.hh_client_secret is None:
            raise HHAuthError(
                "Нет доступа к API hh: не заданы HH_CLIENT_ID и HH_CLIENT_SECRET.\n"
                "Зарегистрируйте приложение на https://dev.hh.ru/admin, скопируйте\n"
                "client_id и client_secret в .env — см. .env.example."
            )

        payload = {
            "grant_type": "client_credentials",
            "client_id": settings.hh_client_id.get_secret_value(),
            "client_secret": settings.hh_client_secret.get_secret_value(),
        }
        client = self._client or httpx.Client(timeout=settings.hh_timeout_seconds)
        try:
            response = client.post(
                f"{settings.hh_api_base}{TOKEN_URL_PATH}",
                data=payload,
                headers={
                    "HH-User-Agent": settings.hh_user_agent,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
        finally:
            if self._client is None:
                client.close()

        if response.status_code != httpx.codes.OK:
            raise HHAuthError(self._explain_failure(response))

        data = response.json()
        try:
            token = data["access_token"]
            expires_in = int(data["expires_in"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HHAuthError(f"hh вернул неожиданный ответ на /token: {data!r}") from exc

        bundle = TokenBundle(
            access_token=token,
            expires_at=datetime.now(UTC) + timedelta(seconds=expires_in),
        )
        logger.info("получен токен приложения, действителен до %s", bundle.expires_at.isoformat())
        return bundle

    @staticmethod
    def _explain_failure(response: httpx.Response) -> str:
        """Превратить ответ hh в сообщение, по которому понятно, что чинить."""
        try:
            body = response.json()
        except ValueError:
            body = {"raw": response.text[:400]}

        error = body.get("error") if isinstance(body, dict) else None
        if error == "invalid_client":
            return (
                "hh отклонил client_id/client_secret (invalid_client).\n"
                "Проверьте пару в .env и то, что приложение на dev.hh.ru не удалено."
            )
        return f"hh вернул {response.status_code} на запрос токена: {body!r}"

    @staticmethod
    def _read_cache(path: Path) -> TokenBundle | None:
        if not path.exists():
            return None
        try:
            return TokenBundle.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError):
            # Битый кэш — не повод падать, просто сходим за новым токеном.
            logger.warning("кэш токена повреждён, будет запрошен новый: %s", path)
            return None

    @staticmethod
    def _write_cache(path: Path, bundle: TokenBundle) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(bundle.to_dict()), encoding="utf-8")
        except OSError as exc:  # pragma: no cover - права на запись, редкий случай
            logger.warning("не удалось сохранить кэш токена (%s), продолжаем без него", exc)
