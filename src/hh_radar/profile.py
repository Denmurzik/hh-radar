"""Профиль кандидата: загрузка из YAML и сопоставление с вакансией.

Смысл модуля объяснён в profile.example.yaml: инструмент должен уметь честно
сказать «сюда не проходишь и вот почему», а не только выдавать список вакансий.
Поэтому ``match_vacancy`` не подкручивает score в сторону оптимизма — при
наличии жёсткого несовпадения (blocker) вердикт всегда "no" независимо от
того, насколько хорошо покрыты навыки.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

from hh_radar.config import get_settings
from hh_radar.db.queries import EXPERIENCE_ORDER
from hh_radar.hh.parse import normalize_skill_name

if TYPE_CHECKING:
    from hh_radar.db.queries import VacancyDetail


class ProfileError(Exception):
    """Базовая ошибка профиля кандидата."""


class ProfileNotFoundError(ProfileError):
    """Файл profile.yaml не найден."""


class ProfileFormatError(ProfileError):
    """Файл profile.yaml найден, но не разобрался или не прошёл валидацию."""


def normalize_skill(raw: str) -> str:
    """Нормализует название навыка для сопоставления с профилем.

    Тонкая обёртка над ``hh_radar.hh.parse.normalize_skill_name`` — там же
    формула и объяснена. Отдельная реализация здесь означала бы, что один и
    тот же навык, попавший в базу через ingest и введённый в profile.yaml
    руками, может нормализоваться по-разному, и match_vacancy молча перестанет
    находить совпадения. Имя и сигнатуру этой обёртки менять нельзя: на неё
    рассчитывает остальной код, обращающийся именно к hh_radar.profile.
    """
    return normalize_skill_name(raw)


@dataclass(frozen=True, slots=True)
class ProfileConstraints:
    """Жёсткие ограничения кандидата. Нарушение любого — blocker в match_vacancy."""

    salary_min_rub: int | None
    remote_only: bool
    relocation: bool
    employment: tuple[str, ...]
    min_experience_accepted: str | None
    english_speaking: bool


@dataclass(frozen=True, slots=True)
class CandidateProfile:
    """Профиль кандидата, загруженный из profile.yaml."""

    title: str
    experience_years: float
    #: Нормализованное имя навыка -> уровень (strong | working | basic).
    skills: dict[str, str]
    #: Технологии, которых у кандидата явно нет (нормализованные имена).
    missing: frozenset[str]
    constraints: ProfileConstraints
    red_flags: tuple[str, ...]
    green_flags: tuple[str, ...]


def load_profile(path: Path | None = None) -> CandidateProfile:
    """Загружает и валидирует профиль кандидата из YAML.

    ``path`` по умолчанию берётся из настроек (``PROFILE_PATH``, обычно
    ``profile.yaml`` в корне проекта). Кидает ``ProfileNotFoundError``, если
    файла нет, и ``ProfileFormatError``, если YAML битый или не соответствует
    ожидаемой структуре — обе ошибки несут понятный русский текст, потому что
    их видит либо человек в терминале, либо агент через MCP-инструмент.
    """
    target = path if path is not None else get_settings().profile_path
    if not target.exists():
        raise ProfileNotFoundError(
            f"Файл профиля не найден: {target}. "
            "Скопируйте profile.example.yaml в profile.yaml и заполните своими данными."
        )

    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ProfileFormatError(f"Не удалось разобрать YAML профиля {target}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ProfileFormatError(f"Профиль {target} должен быть YAML-словарём верхнего уровня")

    try:
        skills = {
            normalize_skill(str(item["name"])): str(item["level"])
            for item in (raw.get("skills") or [])
        }
        constraints_raw: dict[str, Any] = raw.get("constraints") or {}
        constraints = ProfileConstraints(
            salary_min_rub=constraints_raw.get("salary_min_rub"),
            remote_only=bool(constraints_raw.get("remote_only", False)),
            relocation=bool(constraints_raw.get("relocation", False)),
            employment=tuple(constraints_raw.get("employment") or []),
            min_experience_accepted=constraints_raw.get("min_experience_accepted"),
            english_speaking=bool(constraints_raw.get("english_speaking", False)),
        )
        return CandidateProfile(
            title=str(raw["title"]),
            experience_years=float(raw["experience_years"]),
            skills=skills,
            missing=frozenset(normalize_skill(str(s)) for s in (raw.get("missing") or [])),
            constraints=constraints,
            red_flags=tuple(str(s) for s in (raw.get("red_flags") or [])),
            green_flags=tuple(str(s) for s in (raw.get("green_flags") or [])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProfileFormatError(f"Некорректный формат профиля {target}: {exc}") from exc


#: Вес уровня владения навыком в формуле score (см. docstring match_vacancy).
_LEVEL_WEIGHT: dict[str, float] = {"strong": 1.0, "working": 0.7, "basic": 0.4}

#: Доля покрытых ключевых навыков вакансии, начиная с которой verdict "good"
#: вместо "stretch" (при отсутствии blockers).
_GOOD_COVERAGE_THRESHOLD = 0.6


@dataclass(frozen=True, slots=True)
class ProfileMatch:
    """Результат сопоставления вакансии с профилем кандидата."""

    verdict: Literal["good", "stretch", "no"]
    score: float
    matched_skills: list[str]
    missing_skills: list[str]
    blockers: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "score": round(self.score, 3),
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "blockers": self.blockers,
            "notes": self.notes,
        }


def _contains_flag(haystack: str, flag: str) -> bool:
    """Ищет флаг как подстроку без учёта регистра — red/green flags не токены,
    а свободный текст ("выезд в офис"), поэтому полноценный поиск по словам
    здесь избыточен и только усложнил бы код без пользы."""
    return flag.strip().lower() in haystack


def match_vacancy(profile: CandidateProfile, vacancy: VacancyDetail) -> ProfileMatch:
    """Сопоставляет вакансию с профилем кандидата.

    Вердикт:
      - "no" — есть хотя бы один blocker (зарплата ниже минимума, требуемый
        опыт выше принимаемого, не подходящая занятость, вакансия не
        удалённая при remote_only, встретился стоп-слово из red_flags);
      - "good" — блокеров нет, вакансия перечисляет навыки и покрыто ≥60%
        из них;
      - "stretch" — блокеров нет, но покрыто меньше 60%, а также всегда,
        когда вакансия вообще не перечисляет key_skills: сравнивать не с
        чем, и "good" в этом случае означал бы совпадение, которое на самом
        деле не проверялось — только незнание, а не факт.

    Формула score (0..1) — сознательно простая и не зависит от blockers,
    чтобы отвечать на отдельный вопрос "насколько технически близка вакансия",
    а не дублировать verdict:

        score = 0.7 * coverage + 0.3 * avg_level

    где ``coverage`` — доля ключевых навыков вакансии, которые есть у
    кандидата (1.0, если вакансия не указывает навыков вовсе — считать нечего),
    а ``avg_level`` — средний вес уровня владения (strong=1.0, working=0.7,
    basic=0.4) по совпавшим навыкам (0, если совпадений нет). Вес 0.7/0.3
    отражает то, что "есть ли вообще нужный навык" важнее, чем "насколько
    глубоко он изучен" — на старте карьеры basic в требуемом навыке всё ещё
    сильно лучше, чем его полное отсутствие.
    """
    blockers: list[str] = []
    notes: list[str] = []
    constraints = profile.constraints

    if constraints.salary_min_rub is not None:
        if vacancy.salary_from_rub is not None:
            if vacancy.salary_from_rub < constraints.salary_min_rub:
                blockers.append(
                    f"зарплата от {vacancy.salary_from_rub:,} ₽ ниже минимума "
                    f"{constraints.salary_min_rub:,} ₽".replace(",", " ")
                )
        elif vacancy.salary_to_rub is not None:
            # "от" не указано, но и "вилка до" уже ниже минимума — тоже блокер:
            # выше вакансия предложить не может по определению.
            if vacancy.salary_to_rub < constraints.salary_min_rub:
                blockers.append(
                    f"зарплата до {vacancy.salary_to_rub:,} ₽ ниже минимума "
                    f"{constraints.salary_min_rub:,} ₽".replace(",", " ")
                )
        else:
            notes.append("зарплата не указана в вакансии — проверьте условия самостоятельно")

    if constraints.remote_only and not vacancy.is_remote:
        blockers.append("не удалённая работа")

    if (
        constraints.employment
        and vacancy.employment_id is not None
        and vacancy.employment_id not in constraints.employment
    ):
        blockers.append(
            f"тип занятости «{vacancy.employment_id}» не входит в принимаемые "
            f"({', '.join(constraints.employment)})"
        )

    if constraints.min_experience_accepted is not None and vacancy.experience_id is not None:
        required_rank = EXPERIENCE_ORDER.get(vacancy.experience_id)
        accepted_rank = EXPERIENCE_ORDER.get(constraints.min_experience_accepted)
        if (
            required_rank is not None
            and accepted_rank is not None
            and required_rank > accepted_rank
        ):
            blockers.append(
                f"требуется опыт «{vacancy.experience_name or vacancy.experience_id}», "
                "это выше принимаемого уровня"
            )

    haystack = f"{vacancy.name}\n{vacancy.description or ''}".lower()
    for flag in profile.red_flags:
        if _contains_flag(haystack, flag):
            blockers.append(f"в описании встречается стоп-слово: {flag}")
    for flag in profile.green_flags:
        if _contains_flag(haystack, flag):
            notes.append(f"зелёный флаг: {flag}")

    matched_skills = [name for name in vacancy.skills if name in profile.skills]
    missing_skills = [name for name in vacancy.skills if name not in profile.skills]

    for name in missing_skills:
        if name in profile.missing:
            notes.append(f"кандидат явно отмечает пробел в навыке: {name}")

    if vacancy.skills:
        coverage = len(matched_skills) / len(vacancy.skills)
    else:
        # 1.0 — не "полное совпадение", а "сравнивать не с чем": пустая
        # coverage=1.0 нужна только формуле score ниже. verdict эту
        # неоднозначность различает отдельно (см. has_named_skills ниже),
        # иначе отсутствие данных выглядело бы как подтверждённое совпадение.
        coverage = 1.0
        notes.append("навыки в вакансии не перечислены — совпадение проверить не по чему")

    if matched_skills:
        avg_level = sum(
            _LEVEL_WEIGHT.get(profile.skills[name], 0.0) for name in matched_skills
        ) / len(matched_skills)
    else:
        avg_level = 0.0

    score = max(0.0, min(1.0, 0.7 * coverage + 0.3 * avg_level))

    if blockers:
        verdict: Literal["good", "stretch", "no"] = "no"
    elif coverage >= _GOOD_COVERAGE_THRESHOLD and vacancy.skills:
        verdict = "good"
    else:
        verdict = "stretch"

    return ProfileMatch(
        verdict=verdict,
        score=score,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        blockers=blockers,
        notes=notes,
    )
