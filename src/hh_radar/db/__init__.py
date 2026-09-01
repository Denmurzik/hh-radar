"""Слой доступа к данным."""

from hh_radar.db.models import (
    Base,
    Employer,
    Skill,
    Vacancy,
    VacancyChunk,
    vacancy_skills,
)

__all__ = [
    "Base",
    "Employer",
    "Skill",
    "Vacancy",
    "VacancyChunk",
    "vacancy_skills",
]
