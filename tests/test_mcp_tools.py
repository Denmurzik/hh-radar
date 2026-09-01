"""Тесты MCP-инструментов hh-radar.

Работа с базой замокана целиком: инструменты — обычные функции (декоратор
``@server.tool`` возвращает исходный callable, не оборачивает его), поэтому
их можно звать напрямую, подменив ``ping``, ``session_scope`` и функции
``hh_radar.db.queries`` в модуле сервера через monkeypatch.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from hh_radar.db.queries import MAX_LIMIT
from hh_radar.mcp_server import server as server_module
from hh_radar.mcp_server.server import build_server, server

TOOL_NAMES = {
    "search_vacancies",
    "get_vacancy",
    "skill_stats",
    "market_overview",
    "compare_to_profile",
    "semantic_search",
    "db_status",
}


@contextmanager
def _fake_session_scope() -> Any:
    yield MagicMock(name="fake-session")


@pytest.fixture(autouse=True)
def _db_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """По умолчанию база «доступна» — иначе тесты клампа/маппинга не дойдут до логики."""
    monkeypatch.setattr(server_module, "ping", lambda: True)
    monkeypatch.setattr(server_module, "session_scope", _fake_session_scope)


class TestBuildServer:
    """CI-смоук-тест ровно так, как его вызывает пайплайн: build_server().name."""

    def test_build_server_returns_named_server_without_touching_db(self) -> None:
        srv = build_server()
        assert srv.name == "hh-radar"

    def test_build_server_registers_all_seven_tools(self) -> None:
        srv = build_server()
        registered = {tool.name for tool in srv._tool_manager.list_tools()}
        assert registered == TOOL_NAMES

    def test_build_server_returns_a_fresh_instance_each_call(self) -> None:
        assert build_server() is not build_server()


class TestToolRegistry:
    def test_all_seven_tools_are_registered(self) -> None:
        registered = {tool.name for tool in server._tool_manager.list_tools()}
        assert registered == TOOL_NAMES

    def test_every_tool_has_a_non_empty_description(self) -> None:
        for tool in server._tool_manager.list_tools():
            assert tool.description, f"у инструмента {tool.name} пустое описание"
            assert len(tool.description) > 20, f"описание {tool.name} подозрительно короткое"


class TestDbUnavailable:
    def test_search_vacancies_reports_db_unavailable_without_crashing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(server_module, "ping", lambda: False)

        result = server_module.search_vacancies(query="python")

        assert "error" in result
        assert "hint" in result
        assert "docker compose" in result["hint"]


class TestLimitClamping:
    def test_search_vacancies_clamps_limit_before_calling_queries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_search_vacancies(session: Any, query: Any, **kwargs: Any) -> list[Any]:
            captured.update(kwargs)
            return []

        monkeypatch.setattr(server_module.queries, "search_vacancies", fake_search_vacancies)

        server_module.search_vacancies(query="python", limit=10_000)

        assert captured["limit"] == MAX_LIMIT

    def test_skill_stats_clamps_top_n_before_calling_queries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_skill_stats(session: Any, **kwargs: Any) -> list[Any]:
            captured.update(kwargs)
            return []

        monkeypatch.setattr(server_module.queries, "skill_stats", fake_skill_stats)

        server_module.skill_stats(top_n=10_000)

        assert captured["top_n"] == MAX_LIMIT

    def test_semantic_search_clamps_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def fake_semantic_search(session: Any, query: Any, **kwargs: Any) -> list[Any]:
            captured.update(kwargs)
            return []

        # semantic_search импортирует hh_radar.rag.search лениво внутри функции —
        # подменяем сам модуль в sys.modules, чтобы не зависеть от того,
        # существует ли он на диске у коллеги из rag на момент прогона тестов.
        import sys
        import types

        fake_module = types.ModuleType("hh_radar.rag.search")
        fake_module.semantic_search = fake_semantic_search  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "hh_radar.rag.search", fake_module)

        server_module.semantic_search(query="аналитика данных", limit=10_000)

        assert captured["limit"] == MAX_LIMIT


class TestSemanticSearchWithoutRagExtra:
    def test_returns_friendly_error_when_rag_module_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        # None в sys.modules — стандартный способ форсировать ImportError
        # на `import hh_radar.rag.search`, не трогая реальный модуль на диске.
        monkeypatch.setitem(sys.modules, "hh_radar.rag.search", None)

        result = server_module.semantic_search(query="аналитика данных")

        assert "error" in result
        assert "extra rag" in result["hint"] or "rag" in result["hint"]

    def test_does_not_touch_the_database_when_rag_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "hh_radar.rag.search", None)
        ping_calls: list[None] = []
        monkeypatch.setattr(server_module, "ping", lambda: ping_calls.append(None) or True)

        server_module.semantic_search(query="что угодно")

        assert ping_calls == []  # до проверки базы дело даже не дошло


class TestCompareToProfile:
    def test_missing_profile_returns_friendly_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hh_radar.profile import ProfileNotFoundError

        def fake_load_profile(path: Any = None) -> Any:
            raise ProfileNotFoundError("Файл профиля не найден: profile.yaml.")

        monkeypatch.setattr(server_module, "load_profile", fake_load_profile)

        result = server_module.compare_to_profile(vacancy_id=1)

        assert "error" in result
        assert "profile" in result["hint"].lower() or "профил" in result["error"].lower()

    def test_vacancy_not_found_returns_friendly_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(server_module, "load_profile", lambda path=None: MagicMock())
        monkeypatch.setattr(server_module.queries, "get_vacancy", lambda session, vacancy_id: None)

        result = server_module.compare_to_profile(vacancy_id=999)

        assert "error" in result
        assert "999" in result["error"]

    def test_custom_profile_path_is_forwarded_to_load_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pathlib import Path

        captured: dict[str, Any] = {}

        def fake_load_profile(path: Path | None = None) -> Any:
            captured["path"] = path
            return MagicMock()

        monkeypatch.setattr(server_module, "load_profile", fake_load_profile)
        monkeypatch.setattr(server_module.queries, "get_vacancy", lambda session, vacancy_id: None)

        server_module.compare_to_profile(vacancy_id=1, profile_path="custom/profile.yaml")

        assert captured["path"] == Path("custom/profile.yaml")

    def test_default_profile_path_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Без явного profile_path в load_profile должен уйти None — она сама
        возьмёт путь из настроек."""
        captured: dict[str, Any] = {}

        def fake_load_profile(path: Any = None) -> Any:
            captured["path"] = path
            return MagicMock()

        monkeypatch.setattr(server_module, "load_profile", fake_load_profile)
        monkeypatch.setattr(server_module.queries, "get_vacancy", lambda session, vacancy_id: None)

        server_module.compare_to_profile(vacancy_id=1)

        assert captured["path"] is None


class TestSearchVacanciesDescriptionTruncation:
    def test_long_description_is_truncated_in_search_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from hh_radar.db.queries import VacancySummary

        long_description = "а" * 1000
        summary = VacancySummary(
            id=1,
            name="Backend",
            employer_name=None,
            area_name=None,
            salary_from_rub=None,
            salary_to_rub=None,
            salary_currency=None,
            experience_name=None,
            is_remote=False,
            published_at=None,
            alternate_url=None,
            description=long_description,
            rank=None,
        )
        monkeypatch.setattr(
            server_module.queries, "search_vacancies", lambda session, query, **kwargs: [summary]
        )

        result = server_module.search_vacancies(query="python")

        assert len(result["results"][0]["description"]) < len(long_description)

    def test_short_description_is_not_touched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hh_radar.db.queries import VacancySummary

        summary = VacancySummary(
            id=1,
            name="Backend",
            employer_name=None,
            area_name=None,
            salary_from_rub=None,
            salary_to_rub=None,
            salary_currency=None,
            experience_name=None,
            is_remote=False,
            published_at=None,
            alternate_url=None,
            description="Коротко.",
            rank=None,
        )
        monkeypatch.setattr(
            server_module.queries, "search_vacancies", lambda session, query, **kwargs: [summary]
        )

        result = server_module.search_vacancies(query="python")

        assert result["results"][0]["description"] == "Коротко."
