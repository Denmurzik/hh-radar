"""Тесты генератора конфигурации для Claude Desktop.

Проверяется главным образом одно: команда не должна портить чужой конфиг.
У человека там уже могут стоять другие MCP-серверы, и потерять их из-за
нашей утилиты — худшее, что она может сделать.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hh_radar.mcp_server.desktop import (
    SERVER_KEY,
    build_entry,
    config_path,
    install,
    render_snippet,
)


class TestBuildEntry:
    def test_uses_current_interpreter_by_default(self) -> None:
        entry = build_entry()
        assert entry["command"].endswith(("python.exe", "python", "python3"))

    def test_runs_the_package_as_a_module(self) -> None:
        assert build_entry()["args"] == ["-m", "hh_radar.mcp_server"]

    def test_profile_path_is_absolute(self) -> None:
        """Claude Desktop стартует из своего каталога — относительный путь там мимо."""
        assert Path(build_entry()["env"]["PROFILE_PATH"]).is_absolute()

    def test_custom_interpreter_is_respected(self, tmp_path: Path) -> None:
        fake = tmp_path / "python.exe"
        assert build_entry(fake)["command"] == str(fake)


class TestSnippet:
    def test_snippet_is_valid_json_with_our_server(self) -> None:
        parsed = json.loads(render_snippet())
        assert SERVER_KEY in parsed["mcpServers"]


class TestInstall:
    def test_creates_file_and_parent_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "Claude" / "claude_desktop_config.json"
        result = install(target)

        assert result.path.exists()
        assert result.backup is None
        assert result.already_present is False
        assert SERVER_KEY in json.loads(target.read_text(encoding="utf-8"))["mcpServers"]

    def test_keeps_other_servers(self, tmp_path: Path) -> None:
        target = tmp_path / "config.json"
        target.write_text(
            json.dumps({"mcpServers": {"filesystem": {"command": "npx"}}}),
            encoding="utf-8",
        )

        install(target)

        servers = json.loads(target.read_text(encoding="utf-8"))["mcpServers"]
        assert set(servers) == {"filesystem", SERVER_KEY}
        assert servers["filesystem"]["command"] == "npx"

    def test_keeps_unrelated_top_level_keys(self, tmp_path: Path) -> None:
        target = tmp_path / "config.json"
        target.write_text(json.dumps({"globalShortcut": "Ctrl+Q"}), encoding="utf-8")

        install(target)

        assert json.loads(target.read_text(encoding="utf-8"))["globalShortcut"] == "Ctrl+Q"

    def test_makes_a_backup_of_the_previous_version(self, tmp_path: Path) -> None:
        target = tmp_path / "config.json"
        original = json.dumps({"mcpServers": {"other": {"command": "x"}}})
        target.write_text(original, encoding="utf-8")

        result = install(target)

        assert result.backup is not None
        assert result.backup.read_text(encoding="utf-8") == original

    def test_reports_replacement_of_an_existing_entry(self, tmp_path: Path) -> None:
        target = tmp_path / "config.json"
        target.write_text(json.dumps({"mcpServers": {SERVER_KEY: {"command": "старое"}}}), "utf-8")

        assert install(target).already_present is True

    def test_refuses_to_overwrite_broken_json(self, tmp_path: Path) -> None:
        """Лучше сказать «почините файл», чем молча стереть чужие настройки."""
        target = tmp_path / "config.json"
        target.write_text("{ это не json", encoding="utf-8")

        with pytest.raises(ValueError, match="невалидный JSON"):
            install(target)

    def test_rejects_mcpservers_of_wrong_shape(self, tmp_path: Path) -> None:
        target = tmp_path / "config.json"
        target.write_text(json.dumps({"mcpServers": ["не объект"]}), encoding="utf-8")

        with pytest.raises(ValueError, match="mcpServers"):
            install(target)

    def test_empty_file_is_treated_as_empty_config(self, tmp_path: Path) -> None:
        target = tmp_path / "config.json"
        target.write_text("   \n", encoding="utf-8")

        install(target)

        assert SERVER_KEY in json.loads(target.read_text(encoding="utf-8"))["mcpServers"]


class TestConfigPath:
    def test_points_at_a_json_file(self) -> None:
        assert config_path().name == "claude_desktop_config.json"
