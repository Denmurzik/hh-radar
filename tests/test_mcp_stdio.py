"""Проверка сервера по настоящему протоколу MCP через stdio.

Все остальные тесты вызывают функции инструментов напрямую. Этот запускает
сервер отдельным процессом и разговаривает с ним ровно так, как это делает
Claude Desktop: JSON-RPC построчно через stdin/stdout.

Ловит то, что не ловится ничем другим:

* сервер не стартует из-за ошибки на импорте;
* инструменты объявлены, но не регистрируются в протоколе;
* что-нибудь печатается в stdout мимо протокола — при stdio-транспорте
  это ломает разбор сообщений, и клиент видит не ошибку, а молчащий
  сервер. Отдельной проверки на это нет и не нужно: любая посторонняя
  строка в stdout уронит разбор ответа в первом же запросе.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOOLS = {
    "search_vacancies",
    "get_vacancy",
    "skill_stats",
    "market_overview",
    "compare_to_profile",
    "semantic_search",
    "db_status",
}


class StdioClient:
    """Минимальный MCP-клиент: ровно столько, сколько нужно тесту."""

    def __init__(self, process: subprocess.Popen[str]) -> None:
        self._process = process
        self._next_id = 0

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._next_id += 1
        self._write({"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params or {}})
        return self._read()

    def notify(self, method: str) -> None:
        self._write({"jsonrpc": "2.0", "method": method})

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        response = self.request("tools/call", {"name": name, "arguments": arguments})
        payload = response["result"]["content"][0]["text"]
        return json.loads(payload)  # type: ignore[no-any-return]

    def _write(self, message: dict[str, Any]) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(message) + "\n")
        self._process.stdin.flush()

    def _read(self) -> dict[str, Any]:
        assert self._process.stdout is not None
        line = self._process.stdout.readline()
        if not line.strip():
            raise AssertionError("сервер закрыл stdout, не ответив")
        return json.loads(line)  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def server() -> Iterator[StdioClient]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("EMBEDDING_BACKEND", "hash")

    process = subprocess.Popen(
        [sys.executable, "-m", "hh_radar.mcp_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    client = StdioClient(process)
    client.request(
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "1"},
        },
    )
    client.notify("notifications/initialized")
    try:
        yield client
    finally:
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            process.kill()


class TestHandshake:
    def test_server_introduces_itself_with_a_version(self) -> None:
        """Пустая версия в ответе на initialize выглядит как недоделанный сервер."""
        from hh_radar import __version__
        from hh_radar.mcp_server.server import build_server

        built = build_server()
        assert built.name == "hh-radar"
        assert built.version == __version__


class TestProtocol:
    def test_lists_every_tool(self, server: StdioClient) -> None:
        tools = server.request("tools/list")["result"]["tools"]
        assert {tool["name"] for tool in tools} == EXPECTED_TOOLS

    def test_every_tool_explains_itself_to_the_model(self, server: StdioClient) -> None:
        """Плохое описание — главная причина, по которой агент зовёт не тот инструмент."""
        tools = server.request("tools/list")["result"]["tools"]
        for tool in tools:
            description = (tool.get("description") or "").strip()
            assert len(description) > 80, f"{tool['name']}: описание слишком короткое"

    def test_every_tool_declares_an_input_schema(self, server: StdioClient) -> None:
        tools = server.request("tools/list")["result"]["tools"]
        for tool in tools:
            assert tool.get("inputSchema", {}).get("type") == "object", tool["name"]

    def test_db_status_answers_with_real_numbers(self, server: StdioClient) -> None:
        status = server.call_tool("db_status", {})
        assert "vacancies_total" in status
        assert isinstance(status["vacancies_total"], int)

    def test_search_returns_a_result_envelope(self, server: StdioClient) -> None:
        found = server.call_tool("search_vacancies", {"query": "n8n", "limit": 3})
        assert "results" in found or "error" in found

    def test_limit_is_capped_over_the_protocol(self, server: StdioClient) -> None:
        """Ограничение сверху должно работать и через протокол, а не только в питоне."""
        found = server.call_tool("search_vacancies", {"query": "", "limit": 10_000})
        assert "error" not in found
        assert len(found.get("results", [])) <= 50

    def test_unknown_vacancy_is_a_message_not_a_crash(self, server: StdioClient) -> None:
        answer = server.call_tool("get_vacancy", {"vacancy_id": 1})
        assert "error" in answer
        # Сервер обязан пережить ошибку и продолжить обслуживать запросы.
        assert "vacancies_total" in server.call_tool("db_status", {})
