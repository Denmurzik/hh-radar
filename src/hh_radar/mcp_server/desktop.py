"""Подключение сервера к Claude Desktop.

Ручное редактирование ``claude_desktop_config.json`` — самое частое место, где
люди спотыкаются при первом знакомстве с MCP: путь к интерпретатору на Windows
надо экранировать, файла может не быть вовсе, а если он есть — в нём уже лежат
чужие серверы, которые нельзя затирать.

Поэтому конфиг собирается кодом, а запись делается только по явному флагу и
всегда с резервной копией.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hh_radar.config import PROJECT_ROOT, get_settings

SERVER_KEY = "hh-radar"


@dataclass(frozen=True, slots=True)
class ConfigWriteResult:
    path: Path
    written: bool
    backup: Path | None
    already_present: bool


def config_path() -> Path:
    """Где Claude Desktop держит свой конфиг на текущей системе."""
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "Claude" / "claude_desktop_config.json"
    if system == "Darwin":
        mac = Path.home() / "Library" / "Application Support" / "Claude"
        return mac / "claude_desktop_config.json"
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def build_entry(python: Path | None = None) -> dict[str, Any]:
    """Секция для нашего сервера.

    Берётся интерпретатор, из которого команду и запустили: если человек
    работает в виртуальном окружении проекта, именно его путь и нужен —
    системный python пакет не увидит.
    """
    interpreter = python or Path(sys.executable)
    settings = get_settings()
    return {
        "command": str(interpreter),
        "args": ["-m", "hh_radar.mcp_server"],
        "env": {
            "DATABASE_URL": settings.database_url,
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            # Абсолютный путь обязателен: Claude Desktop запускает сервер
            # из своего рабочего каталога, и относительный "profile.yaml"
            # там указывает в пустоту.
            "PROFILE_PATH": str(settings.profile_path.expanduser().resolve()),
        },
    }


def render_snippet() -> str:
    """Готовый кусок JSON, который можно вставить руками."""
    return json.dumps({"mcpServers": {SERVER_KEY: build_entry()}}, indent=2, ensure_ascii=False)


def install(path: Path | None = None, *, python: Path | None = None) -> ConfigWriteResult:
    """Вписать сервер в конфиг Claude Desktop, сохранив всё остальное."""
    target = path or config_path()
    existing: dict[str, Any] = {}
    backup: Path | None = None

    if target.exists():
        raw = target.read_text(encoding="utf-8").strip()
        if raw:
            try:
                existing = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{target} содержит невалидный JSON ({exc}). "
                    "Почините файл или удалите его — перезаписывать вслепую не буду."
                ) from exc
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        backup = target.with_suffix(f".{stamp}.bak")
        backup.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")

    servers = existing.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise ValueError(f"{target}: поле mcpServers не объект, чинить руками")

    already = SERVER_KEY in servers
    servers[SERVER_KEY] = build_entry(python)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return ConfigWriteResult(path=target, written=True, backup=backup, already_present=already)
