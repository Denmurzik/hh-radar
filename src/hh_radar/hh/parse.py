"""Превращение сырого JSON от hh в плоские записи для базы.

Модуль намеренно чистый: ни сети, ни базы, ни глобального состояния. Всё, что
здесь есть, — функции «словарь на входе, датакласс на выходе». Именно поэтому
он покрыт тестами плотнее остального кода: это единственное место, где ломается
сборщик, когда hh меняет формат ответа.

Три вещи, о которых стоит знать читателю:

* hh отдаёт зарплату в валюте вакансии и кодирует рубли как ``RUR``, а не
  ``RUB``. Для сортировки нужна общая шкала, поэтому есть приведение к рублям
  по статичной таблице курсов. Это приближение, и оно помечено как приближение
  в README: тянуть курсы ЦБ ради сортировки вакансий — усложнение без пользы.
* Поле зарплаты в разное время называлось ``salary`` и ``salary_range``.
  Поддержаны оба: старое как запасной вариант.
* Удалённость вакансии приходится выводить из двух полей — ``work_format``
  и ``schedule``, — потому что hh переносил этот признак между ними и в
  выдаче встречаются оба варианта.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

#: Приближённые курсы к рублю. Ровно для одной цели — сортировать и фильтровать
#: вакансии по «примерно сколько это в рублях». Точность здесь не нужна и
#: сознательно принесена в жертву отсутствию ещё одной внешней зависимости.
CURRENCY_RATES: dict[str, float] = {
    "RUR": 1.0,
    "RUB": 1.0,
    "USD": 90.0,
    "EUR": 98.0,
    "KZT": 0.19,
    "BYR": 27.0,
    "BYN": 27.0,
    "UAH": 2.2,
    "UZS": 0.007,
    "KGS": 1.03,
    "AZN": 53.0,
    "GEL": 33.0,
}

#: Значения work_format / schedule, означающие удалённую работу.
_REMOTE_MARKERS = {"remote", "удаленная работа", "удалённая работа", "удалённо", "удаленно"}

_WHITESPACE = re.compile(r"\s+")
_EDGE_PUNCT = re.compile(r"^[\s\-–—•·,.;:/\\()\[\]{}\"'«»]+|[\s\-–—•·,.;:/\\()\[\]{}\"'«»]+$")


@dataclass(frozen=True, slots=True)
class ParsedEmployer:
    id: int
    name: str
    alternate_url: str | None = None
    trusted: bool = False


@dataclass(frozen=True, slots=True)
class ParsedVacancy:
    """Вакансия в том виде, в котором она ложится в таблицу."""

    id: int
    name: str
    employer: ParsedEmployer | None = None
    area_id: int | None = None
    area_name: str | None = None
    salary_from: int | None = None
    salary_to: int | None = None
    salary_currency: str | None = None
    salary_gross: bool | None = None
    salary_from_rub: int | None = None
    salary_to_rub: int | None = None
    experience_id: str | None = None
    experience_name: str | None = None
    employment_id: str | None = None
    schedule_id: str | None = None
    is_remote: bool = False
    professional_roles: list[dict[str, Any]] = field(default_factory=list)
    description: str | None = None
    alternate_url: str | None = None
    published_at: datetime | None = None
    created_at: datetime | None = None
    archived: bool = False
    skills: list[str] = field(default_factory=list)
    #: True, если запись собрана из полной карточки, а не из поисковой выдачи.
    is_detailed: bool = False


# --------------------------------------------------------------------- api --


def parse_vacancy(raw: dict[str, Any]) -> ParsedVacancy:
    """Разобрать вакансию из поиска или из полной карточки.

    Отличать одно от другого не нужно: полная карточка — это надмножество,
    и признак ``is_detailed`` выводится по наличию ``description``.
    """
    salary = _pick_salary(raw)
    currency = _as_str(salary.get("currency"))
    salary_from = _as_int(salary.get("from"))
    salary_to = _as_int(salary.get("to"))

    return ParsedVacancy(
        id=int(raw["id"]),
        name=_as_str(raw.get("name")) or "",
        employer=parse_employer(raw.get("employer")),
        area_id=_as_int(_nested(raw, "area", "id")),
        area_name=_as_str(_nested(raw, "area", "name")),
        salary_from=salary_from,
        salary_to=salary_to,
        salary_currency=currency,
        salary_gross=_as_bool(salary.get("gross")),
        salary_from_rub=to_rub(salary_from, currency),
        salary_to_rub=to_rub(salary_to, currency),
        experience_id=_as_str(_nested(raw, "experience", "id")),
        experience_name=_as_str(_nested(raw, "experience", "name")),
        employment_id=_as_str(_nested(raw, "employment", "id")),
        schedule_id=_as_str(_nested(raw, "schedule", "id")),
        is_remote=detect_remote(raw),
        professional_roles=_as_role_list(raw.get("professional_roles")),
        description=_as_str(raw.get("description")),
        alternate_url=_as_str(raw.get("alternate_url")),
        published_at=parse_datetime(raw.get("published_at")),
        created_at=parse_datetime(raw.get("created_at")),
        archived=bool(raw.get("archived", False)),
        skills=parse_key_skills(raw.get("key_skills")),
        is_detailed=raw.get("description") is not None,
    )


def parse_employer(raw: Any) -> ParsedEmployer | None:
    """Работодатель может быть null: hh так помечает вакансии от физлиц."""
    if not isinstance(raw, dict):
        return None
    employer_id = _as_int(raw.get("id"))
    name = _as_str(raw.get("name"))
    if employer_id is None or not name:
        return None
    return ParsedEmployer(
        id=employer_id,
        name=name,
        alternate_url=_as_str(raw.get("alternate_url")),
        trusted=bool(raw.get("trusted", False)),
    )


def parse_key_skills(raw: Any) -> list[str]:
    """``key_skills`` приходит списком объектов ``{"name": "Python"}``.

    Дубликаты внутри одной вакансии встречаются (работодатель написал
    «Python» и «python»), поэтому схлопываем по нормализованной форме,
    сохраняя порядок — он несёт смысл, первым идёт главное требование.
    """
    if not isinstance(raw, list):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for item in raw:
        name = item.get("name") if isinstance(item, dict) else item
        if not isinstance(name, str):
            continue
        cleaned = name.strip()
        if not cleaned:
            continue
        key = cleaned.casefold().replace("ё", "е")
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def parse_datetime(raw: Any) -> datetime | None:
    """hh отдаёт даты в ISO 8601 со смещением вида ``+0300`` без двоеточия.

    ``datetime.fromisoformat`` в Python 3.11+ такое понимает, но на всякий
    случай нормализуем смещение — формат в ответах hh исторически плавал.
    """
    if not isinstance(raw, str) or not raw:
        return None
    candidate = raw.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    match = re.search(r"([+-]\d{2})(\d{2})$", candidate)
    if match:
        candidate = f"{candidate[: match.start()]}{match.group(1)}:{match.group(2)}"
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None


def to_rub(amount: int | None, currency: str | None) -> int | None:
    """Привести сумму к рублям по статичному курсу. None остаётся None."""
    if amount is None:
        return None
    rate = CURRENCY_RATES.get((currency or "RUR").upper())
    if rate is None:
        # Незнакомая валюта: лучше не показать зарплату, чем показать неверную.
        return None
    return round(amount * rate)


def detect_remote(raw: dict[str, Any]) -> bool:
    """Удалёнка размазана по двум полям — проверяем оба."""
    work_format = raw.get("work_format")
    if isinstance(work_format, list):
        for item in work_format:
            if isinstance(item, dict) and _looks_remote(item.get("id"), item.get("name")):
                return True

    schedule = raw.get("schedule")
    return isinstance(schedule, dict) and _looks_remote(schedule.get("id"), schedule.get("name"))


def normalize_skill_name(raw: str) -> str:
    """Нормализованная форма навыка для агрегации.

    Живёт здесь, а не в profile.py, чтобы parse-слой не зависел от слоя
    профиля; ``hh_radar.profile.normalize_skill`` делегирует сюда.
    """
    cleaned = _EDGE_PUNCT.sub("", raw)
    cleaned = _WHITESPACE.sub(" ", cleaned)
    return cleaned.casefold().replace("ё", "е").strip()


# --------------------------------------------------------------- internals --


def _pick_salary(raw: dict[str, Any]) -> dict[str, Any]:
    """Новое поле ``salary_range`` в приоритете, старое ``salary`` — запасное."""
    for key in ("salary_range", "salary"):
        value = raw.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _looks_remote(*values: Any) -> bool:
    return any(
        isinstance(value, str) and value.strip().casefold() in _REMOTE_MARKERS for value in values
    )


def _nested(raw: dict[str, Any], *path: str) -> Any:
    current: Any = raw
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _as_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _as_role_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
