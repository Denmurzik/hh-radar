"""Тесты против настоящего PostgreSQL.

Существуют по конкретному поводу. Тесты, проверяющие только скомпилированный
SQL, пропустили выражение ``avg(CAST(is_remote AS FLOAT))``: строка запроса
собиралась корректно, а Postgres на ней падал с ``cannot cast type boolean to
double precision``. Такие вещи ловятся только выполнением.

Каждый тест работает в транзакции, которая откатывается — база после прогона
остаётся такой же, какой была.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from hh_radar.db import queries
from hh_radar.hh.parse import parse_vacancy
from hh_radar.ingest.pipeline import IngestReport, link_skills, upsert_vacancy

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def seeded(db_session: Session) -> Session:
    """Положить в базу примеры и вернуть ту же сессию."""
    report = IngestReport()
    for name in ("vacancy_search_page.json", "vacancy_detail.json"):
        payload = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
        items = payload["items"] if isinstance(payload, dict) and "items" in payload else [payload]
        for raw in items:
            parsed = parse_vacancy(raw)
            upsert_vacancy(db_session, parsed, report, mark_detailed=parsed.is_detailed)
            link_skills(db_session, parsed, report)
    db_session.flush()
    return db_session


class TestUpsert:
    def test_writes_vacancies_and_employers(self, seeded: Session) -> None:
        status = queries.db_status(seeded)
        assert status.vacancies_total == 4
        # У одной вакансии в примерах работодатель null — это нормально.
        assert status.employers_total == 3

    def test_repeated_ingest_does_not_duplicate(self, db_session: Session) -> None:
        raw = json.loads((FIXTURES / "vacancy_detail.json").read_text(encoding="utf-8"))
        parsed = parse_vacancy(raw)
        report = IngestReport()

        upsert_vacancy(db_session, parsed, report, mark_detailed=True)
        upsert_vacancy(db_session, parsed, report, mark_detailed=True)
        db_session.flush()

        assert queries.db_status(db_session).vacancies_total == 1
        assert report.inserted == 1
        assert report.updated == 1

    def test_search_item_does_not_erase_loaded_description(self, db_session: Session) -> None:
        """Выдача поиска приходит без описания и не должна затирать полное."""
        detail = parse_vacancy(
            json.loads((FIXTURES / "vacancy_detail.json").read_text(encoding="utf-8"))
        )
        report = IngestReport()
        upsert_vacancy(db_session, detail, report, mark_detailed=True)
        db_session.flush()

        stripped = json.loads((FIXTURES / "vacancy_detail.json").read_text(encoding="utf-8"))
        del stripped["description"]
        upsert_vacancy(db_session, parse_vacancy(stripped), report, mark_detailed=False)
        db_session.flush()

        stored = queries.get_vacancy(db_session, detail.id)
        assert stored is not None
        assert stored.description


class TestFullTextSearch:
    def test_generated_tsvector_is_filled_by_postgres(self, seeded: Session) -> None:
        found = queries.search_vacancies(seeded, "n8n", limit=10)
        assert [item.id for item in found]

    def test_russian_stemming_works(self, seeded: Session) -> None:
        """«автоматизации» должно находить «автоматизация» — за это отвечает
        словарь russian, а не наш код."""
        assert queries.search_vacancies(seeded, "автоматизации", limit=10)

    def test_empty_query_returns_everything_by_date(self, seeded: Session) -> None:
        found = queries.search_vacancies(seeded, "", limit=10)
        assert len(found) == 4

    def test_filters_combine(self, seeded: Session) -> None:
        remote = queries.search_vacancies(seeded, "", remote_only=True, limit=10)
        assert remote
        assert all(item.is_remote for item in remote)

    def test_salary_filter_uses_normalized_rubles(self, seeded: Session) -> None:
        rich = queries.search_vacancies(seeded, "", salary_min_rub=140000, limit=10)
        assert all(item.salary_from_rub is None or item.salary_from_rub >= 140000 for item in rich)


class TestAggregates:
    def test_market_overview_executes(self, seeded: Session) -> None:
        """Регрессия на приведение boolean: раньше падало на remote_share."""
        overview = queries.market_overview(seeded)
        assert overview.total == 4
        assert 0.0 <= overview.remote_share <= 1.0

    def test_market_overview_percentiles(self, seeded: Session) -> None:
        overview = queries.market_overview(seeded)
        assert overview.with_salary == 3
        assert overview.salary_p50 is not None

    def test_skill_stats_counts_and_shares(self, seeded: Session) -> None:
        stats = queries.skill_stats(seeded, top_n=10)
        assert stats
        assert all(0.0 <= stat.share <= 1.0 for stat in stats)
        # Навыки есть только у одной вакансии из примеров — у полной карточки.
        assert {stat.name for stat in stats} >= {"python", "n8n", "docker"}

    def test_skills_deduplicated_across_case(self, seeded: Session) -> None:
        """«Python» и «python» в одной вакансии — один навык, не два."""
        stats = {stat.name: stat.vacancy_count for stat in queries.skill_stats(seeded, top_n=50)}
        assert stats["python"] == 1


class TestVacancyDetail:
    def test_returns_skills_and_employer(self, seeded: Session) -> None:
        detail = queries.get_vacancy(seeded, 135401648)
        assert detail is not None
        assert detail.employer is not None
        assert detail.employer.name == "Глобус-М"
        assert len(detail.skills) == 4

    def test_missing_vacancy_is_none(self, seeded: Session) -> None:
        assert queries.get_vacancy(seeded, 1) is None
