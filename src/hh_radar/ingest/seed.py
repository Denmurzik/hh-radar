"""Наполнение базы примерами без обращения к API.

Нужно ровно для одного сценария: человек склонировал репозиторий, поднял
docker compose, но токена приложения hh у него нет и заводить его ради
пятиминутного знакомства он не станет. ``hh-radar seed`` даёт ему рабочую
базу, на которой видно и полнотекстовый поиск, и MCP-инструменты.

Примеры лежат в ``samples/`` в том же формате, в котором их отдаёт API, и
проходят через тот же :func:`hh_radar.hh.parse.parse_vacancy`, что и живые
данные. Это не отдельная кодовая ветка «для демо» — иначе демо однажды
разошлось бы с реальностью.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sqlalchemy.orm import Session

from hh_radar.config import PROJECT_ROOT
from hh_radar.hh.parse import parse_vacancy
from hh_radar.ingest.pipeline import IngestReport, link_skills, upsert_vacancy

logger = logging.getLogger(__name__)

DEFAULT_SAMPLES_DIR = PROJECT_ROOT / "samples"


def seed_from_samples(session: Session, samples_dir: Path | None = None) -> IngestReport:
    """Загрузить все примеры из каталога в базу."""
    directory = samples_dir or DEFAULT_SAMPLES_DIR
    report = IngestReport(queries=["samples"])

    if not directory.exists():
        raise FileNotFoundError(f"каталог с примерами не найден: {directory}")

    for path in sorted(directory.glob("*.json")):
        for raw in _iter_vacancies(path):
            parsed = parse_vacancy(raw)
            upsert_vacancy(session, parsed, report, mark_detailed=parsed.is_detailed)
            link_skills(session, parsed, report)
            report.seen += 1
        logger.info("загружено из %s", path.name)

    session.commit()
    return report


def _iter_vacancies(path: Path) -> list[dict]:
    """Файл может быть как страницей поиска, так и одной карточкой или списком."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "items" in payload:
        return [item for item in payload["items"] if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []
