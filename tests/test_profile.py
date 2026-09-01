"""Тесты профиля кандидата: нормализация навыков, загрузка YAML, сопоставление.

Для сценариев match_vacancy вакансии собираются вручную как VacancyDetail —
без базы и без ORM, ровно то, что попадает в этот dataclass после get_vacancy.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hh_radar.db.queries import EmployerInfo, VacancyDetail
from hh_radar.profile import (
    CandidateProfile,
    ProfileConstraints,
    ProfileFormatError,
    ProfileNotFoundError,
    load_profile,
    match_vacancy,
    normalize_skill,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "profile_test.yaml"


class TestNormalizeSkill:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  Python  ", "python"),
            ("PostgreSQL", "postgresql"),
            ("REST   API", "rest api"),
            ("- Docker,", "docker"),
            ("C++", "c++"),
            ("Тестёр", "тестер"),
            ("«n8n»", "n8n"),
        ],
    )
    def test_normalization(self, raw: str, expected: str) -> None:
        assert normalize_skill(raw) == expected


class TestLoadProfile:
    def test_loads_valid_fixture(self) -> None:
        profile = load_profile(FIXTURE_PATH)

        assert profile.title == "Backend-разработчик (тест)"
        assert profile.experience_years == pytest.approx(1.5)
        assert profile.skills == {"python": "strong", "postgresql": "working", "docker": "basic"}
        assert profile.missing == frozenset({"kubernetes", "go"})
        assert profile.constraints.salary_min_rub == 100000
        assert profile.constraints.remote_only is True
        assert profile.constraints.employment == ("full",)
        assert profile.constraints.min_experience_accepted == "between1And3"
        assert profile.red_flags == ("1С", "выезд в офис")
        assert profile.green_flags == ("наставник",)

    def test_missing_file_raises_readable_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "no-such-profile.yaml"

        with pytest.raises(ProfileNotFoundError, match=r"profile\.example\.yaml"):
            load_profile(missing)

    def test_broken_yaml_raises_format_error(self, tmp_path: Path) -> None:
        broken = tmp_path / "profile.yaml"
        broken.write_text("title: [unclosed", encoding="utf-8")

        with pytest.raises(ProfileFormatError):
            load_profile(broken)

    def test_missing_required_field_raises_format_error(self, tmp_path: Path) -> None:
        incomplete = tmp_path / "profile.yaml"
        incomplete.write_text(yaml.safe_dump({"title": "Тест"}), encoding="utf-8")

        with pytest.raises(ProfileFormatError):
            load_profile(incomplete)


def _profile(**overrides: object) -> CandidateProfile:
    defaults: dict[str, object] = {
        "title": "Backend-разработчик",
        "experience_years": 2.0,
        "skills": {"python": "strong", "postgresql": "working", "docker": "working"},
        "missing": frozenset({"kubernetes"}),
        "constraints": ProfileConstraints(
            salary_min_rub=100_000,
            remote_only=True,
            relocation=False,
            employment=("full",),
            min_experience_accepted="between1And3",
            english_speaking=False,
        ),
        "red_flags": ("1С", "выезд в офис"),
        "green_flags": ("наставник",),
    }
    defaults.update(overrides)
    return CandidateProfile(**defaults)  # type: ignore[arg-type]


def _vacancy(**overrides: object) -> VacancyDetail:
    defaults: dict[str, object] = {
        "id": 1,
        "name": "Backend-разработчик",
        "employer": EmployerInfo(id=1, name="Acme", alternate_url=None, trusted=True),
        "area_id": 1,
        "area_name": "Москва",
        "salary_from": 150_000,
        "salary_to": None,
        "salary_currency": "RUR",
        "salary_gross": False,
        "salary_from_rub": 150_000,
        "salary_to_rub": None,
        "experience_id": "between1And3",
        "experience_name": "От 1 года до 3 лет",
        "employment_id": "full",
        "schedule_id": "remote",
        "is_remote": True,
        "professional_roles": None,
        "description": "Пишем на Python и PostgreSQL, без легаси.",
        "alternate_url": "https://hh.ru/vacancy/1",
        "published_at": None,
        "created_at": None,
        "archived": False,
        "skills": ["python", "postgresql"],
    }
    defaults.update(overrides)
    return VacancyDetail(**defaults)  # type: ignore[arg-type]


class TestMatchVacancy:
    def test_good_match_when_skills_covered_and_no_blockers(self) -> None:
        profile = _profile()
        vacancy = _vacancy(skills=["python", "postgresql"])

        result = match_vacancy(profile, vacancy)

        assert result.verdict == "good"
        assert not result.blockers
        assert result.matched_skills == ["python", "postgresql"]
        assert result.missing_skills == []
        assert 0.0 < result.score <= 1.0

    def test_salary_below_minimum_is_a_blocker(self) -> None:
        profile = _profile()
        vacancy = _vacancy(salary_from_rub=60_000, skills=["python", "postgresql"])

        result = match_vacancy(profile, vacancy)

        assert result.verdict == "no"
        assert any("зарплата" in b for b in result.blockers)

    def test_salary_to_below_minimum_is_a_blocker_when_from_is_absent(self) -> None:
        """ "От" не указано, но верхняя граница вилки уже ниже минимума —
        выше вакансия предложить не может по определению."""
        profile = _profile()
        vacancy = _vacancy(
            salary_from_rub=None, salary_to_rub=80_000, skills=["python", "postgresql"]
        )

        result = match_vacancy(profile, vacancy)

        assert result.verdict == "no"
        assert any("зарплата" in b for b in result.blockers)

    def test_stop_word_in_description_is_a_blocker(self) -> None:
        profile = _profile()
        vacancy = _vacancy(description="Стек: 1С, доработка конфигураций.")

        result = match_vacancy(profile, vacancy)

        assert result.verdict == "no"
        assert any("1С" in b for b in result.blockers)

    def test_partial_skill_coverage_is_a_stretch(self) -> None:
        profile = _profile()
        # Python есть у кандидата, kafka и go — явно объявленные пробелы.
        vacancy = _vacancy(skills=["python", "kafka", "go"])

        result = match_vacancy(profile, vacancy)

        assert result.verdict == "stretch"
        assert not result.blockers
        assert result.matched_skills == ["python"]
        assert set(result.missing_skills) == {"kafka", "go"}

    def test_vacancy_without_key_skills_cannot_be_good_even_without_blockers(self) -> None:
        """Пустой vacancy.skills не должен читаться как "все навыки совпали":
        verdict капается на "stretch", даже когда блокеров нет вовсе."""
        profile = _profile()
        vacancy = _vacancy(skills=[])

        result = match_vacancy(profile, vacancy)

        assert result.verdict == "stretch"
        assert not result.blockers
        assert result.matched_skills == []
        assert result.missing_skills == []
        assert any("не перечислены" in note for note in result.notes)

    def test_not_remote_is_a_blocker_when_remote_required(self) -> None:
        profile = _profile()
        vacancy = _vacancy(is_remote=False)

        result = match_vacancy(profile, vacancy)

        assert result.verdict == "no"
        assert "не удалённая работа" in result.blockers

    def test_score_does_not_collapse_to_zero_just_because_of_a_blocker(self) -> None:
        """Score отражает техническое соответствие независимо от verdict —
        иначе он не даёт полезной информации при отказе по формальному пункту."""
        profile = _profile()
        vacancy = _vacancy(salary_from_rub=1, skills=["python", "postgresql"])

        result = match_vacancy(profile, vacancy)

        assert result.verdict == "no"
        assert result.score > 0.5
