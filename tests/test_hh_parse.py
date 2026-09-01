"""Тесты разбора ответов hh.

Разбор — самое хрупкое место сборщика: он ломается не от нашего кода, а от
чужих изменений в чужом API. Поэтому проверяются не только счастливые пути,
но и всё, что hh реально присылает: null вместо объекта зарплаты, работодатель
как null, дубликаты навыков в разном регистре, смещение таймзоны без двоеточия.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from hh_radar.hh.parse import (
    CURRENCY_RATES,
    detect_remote,
    normalize_skill_name,
    parse_datetime,
    parse_employer,
    parse_key_skills,
    parse_vacancy,
    to_rub,
)


class TestParseVacancy:
    def test_detail_maps_every_field(self, vacancy_detail: dict) -> None:
        parsed = parse_vacancy(vacancy_detail)

        assert parsed.id == 135401648
        assert parsed.name.startswith("AI-разработчик")
        assert parsed.area_id == 1
        assert parsed.area_name == "Москва"
        assert parsed.experience_id == "between1And3"
        assert parsed.employment_id == "full"
        assert parsed.schedule_id == "remote"
        assert parsed.alternate_url == "https://hh.ru/vacancy/135401648"
        assert parsed.archived is False
        assert parsed.is_detailed is True

    def test_detail_reads_employer(self, vacancy_detail: dict) -> None:
        employer = parse_vacancy(vacancy_detail).employer
        assert employer is not None
        assert employer.id == 5745219
        assert employer.name == "Глобус-М"
        assert employer.trusted is True

    def test_salary_range_wins_over_legacy_salary(self, vacancy_detail: dict) -> None:
        """Оба поля присутствуют — берём новое, но результат должен совпасть."""
        vacancy_detail["salary"] = {"from": 1, "to": 2, "currency": "RUR", "gross": True}
        parsed = parse_vacancy(vacancy_detail)
        assert parsed.salary_from == 150000
        assert parsed.salary_gross is False

    def test_falls_back_to_legacy_salary_field(self, vacancy_detail: dict) -> None:
        del vacancy_detail["salary_range"]
        parsed = parse_vacancy(vacancy_detail)
        assert parsed.salary_from == 150000
        assert parsed.salary_currency == "RUR"

    def test_missing_salary_is_none_not_zero(self, search_page: dict) -> None:
        """Вакансия без зарплаты — это None. Ноль соврал бы при сортировке."""
        without_salary = next(item for item in search_page["items"] if item["id"] == "135859033")
        parsed = parse_vacancy(without_salary)
        assert parsed.salary_from is None
        assert parsed.salary_from_rub is None

    def test_foreign_currency_converted_for_sorting(self, search_page: dict) -> None:
        usd = next(item for item in search_page["items"] if item["id"] == "999000111")
        parsed = parse_vacancy(usd)
        assert parsed.salary_currency == "USD"
        assert parsed.salary_from == 2000
        assert parsed.salary_from_rub == round(2000 * CURRENCY_RATES["USD"])

    def test_null_employer_survives(self, search_page: dict) -> None:
        anonymous = next(item for item in search_page["items"] if item["id"] == "999000111")
        assert parse_vacancy(anonymous).employer is None

    def test_search_item_is_not_marked_detailed(self, search_page: dict) -> None:
        """У выдачи поиска нет description — значит карточка неполная."""
        parsed = parse_vacancy(search_page["items"][0])
        assert parsed.is_detailed is False
        assert parsed.description is None

    def test_id_arrives_as_string_but_stored_as_int(self, search_page: dict) -> None:
        assert parse_vacancy(search_page["items"][0]).id == 136048682


class TestKeySkills:
    def test_deduplicates_ignoring_case(self, vacancy_detail: dict) -> None:
        skills = parse_key_skills(vacancy_detail["key_skills"])
        assert skills == ["Python", "n8n", "REST API", "Docker"]

    def test_missing_field_is_empty_list(self) -> None:
        assert parse_key_skills(None) == []

    def test_garbage_entries_are_skipped(self) -> None:
        assert parse_key_skills([{"name": ""}, {"nope": 1}, 42, {"name": "Go"}]) == ["Go"]

    def test_yo_and_e_are_the_same_skill(self) -> None:
        assert parse_key_skills([{"name": "Обучение"}, {"name": "Обучение"}]) == ["Обучение"]


class TestParseDatetime:
    def test_offset_without_colon(self) -> None:
        parsed = parse_datetime("2026-08-24T11:03:17+0300")
        assert parsed is not None
        assert parsed.utcoffset() is not None
        assert parsed.astimezone(UTC) == datetime(2026, 8, 24, 8, 3, 17, tzinfo=UTC)

    def test_zulu_suffix(self) -> None:
        assert parse_datetime("2026-08-24T08:03:17Z") == datetime(2026, 8, 24, 8, 3, 17, tzinfo=UTC)

    @pytest.mark.parametrize("value", [None, "", "не дата", 12345, {"a": 1}])
    def test_garbage_is_none_not_exception(self, value: object) -> None:
        assert parse_datetime(value) is None


class TestToRub:
    @pytest.mark.parametrize(
        ("amount", "currency", "expected"),
        [
            (100000, "RUR", 100000),
            (100000, "RUB", 100000),
            (1000, "USD", 90000),
            (1000, "usd", 90000),
            (None, "USD", None),
        ],
    )
    def test_conversion(self, amount: int | None, currency: str, expected: int | None) -> None:
        assert to_rub(amount, currency) == expected

    def test_unknown_currency_yields_none(self) -> None:
        """Лучше не показать зарплату, чем показать неверную."""
        assert to_rub(1000, "XYZ") is None

    def test_missing_currency_treated_as_rubles(self) -> None:
        assert to_rub(1000, None) == 1000


class TestDetectRemote:
    def test_from_work_format(self) -> None:
        assert detect_remote({"work_format": [{"id": "REMOTE", "name": "Удалённо"}]}) is True

    def test_from_schedule_when_work_format_empty(self) -> None:
        assert (
            detect_remote(
                {"work_format": [], "schedule": {"id": "remote", "name": "Удаленная работа"}}
            )
            is True
        )

    def test_office_vacancy(self) -> None:
        assert detect_remote({"schedule": {"id": "fullDay", "name": "Полный день"}}) is False

    def test_absent_fields(self) -> None:
        assert detect_remote({}) is False


class TestNormalizeSkillName:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  Python  ", "python"),
            ("REST   API", "rest api"),
            ("Обучение", "обучение"),
            ("- Docker,", "docker"),
            ("«n8n»", "n8n"),
        ],
    )
    def test_normalization(self, raw: str, expected: str) -> None:
        assert normalize_skill_name(raw) == expected


class TestParseEmployer:
    def test_none_input(self) -> None:
        assert parse_employer(None) is None

    def test_employer_without_id_is_dropped(self) -> None:
        assert parse_employer({"name": "Без id"}) is None

    def test_employer_without_name_is_dropped(self) -> None:
        assert parse_employer({"id": "1"}) is None
