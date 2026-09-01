"""Тесты печати готового текста в терминал.

Rich удобен для сообщений, которые пишет сама программа, и вреден для текста,
который она обязана отдать без изменений: квадратные скобки он считает
разметкой и молча съедает, а длинные строки переносит по ширине окна.

Оба случая тут уже случались: из отчёта оценки пропадала пометка метода
``[fulltext]``, а JSON для Claude Desktop разрывался посреди пути к
интерпретатору. Ошибка тихая — вывод выглядит нормально, пока не сравнишь
его с оригиналом, — поэтому она закреплена тестом.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

import hh_radar.cli as cli


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Узкая консоль: на широкой перенос строк не воспроизвёлся бы."""
    buffer = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=buffer, width=40, no_color=True))
    return buffer


def test_square_brackets_survive(captured: io.StringIO) -> None:
    cli.echo_raw("- **q01** [fulltext] «запрос»")
    assert "[fulltext]" in captured.getvalue()


def test_long_line_is_not_wrapped(captured: io.StringIO) -> None:
    line = '"command": "C:\\\\Users\\\\dev\\\\hh-radar\\\\.venv\\\\Scripts\\\\python.exe",'
    cli.echo_raw(line)
    assert captured.getvalue().strip() == line
