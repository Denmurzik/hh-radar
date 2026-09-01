"""Тесты слоя запросов без живой базы.

Два вида тестов:

1. Компиляция SQL — строим Select через приватные билдеры (``_search_statement``,
   ``_apply_market_filters``, ``_skill_stats_statement``) и проверяем текст и
   связанные параметры скомпилированного запроса. ``literal_binds=False``
   (как договорились) — потому что websearch_to_tsquery получает первым
   аргументом REGCONFIG, для которого у SQLAlchemy нет литерального рендера,
   и запрос с literal_binds=True на нём просто падает при компиляции.
2. Маппинг строк в dataclass'ы — сессия подменяется простой заглушкой,
   отдающей заранее заданный результат, и проверяется то, что верхнеуровневые
   функции модуля правильно собирают из него dataclass'ы.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.dialects import postgresql

from hh_radar.db.models import Employer, Skill, Vacancy
from hh_radar.db.queries import (
    MAX_LIMIT,
    DbStatus,
    EmployerInfo,
    MarketOverview,
    VacancySummary,
    _apply_market_filters,
    _search_statement,
    _skill_stats_statement,
    db_status,
    get_vacancy,
    market_overview,
    search_vacancies,
    skill_stats,
)


def _compile(stmt: Select[Any]) -> tuple[str, dict[str, Any]]:
    """Компилирует Select с bind-параметрами и возвращает (SQL, значения параметров)."""
    compiled = stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": False})
    return str(compiled), dict(compiled.params)


class TestSearchStatement:
    def test_empty_query_has_no_fulltext_clause(self) -> None:
        stmt = _search_statement(
            "",
            area_id=None,
            salary_min_rub=None,
            experience_ids=None,
            remote_only=False,
            published_within_days=None,
            limit=20,
            offset=0,
        )
        sql, params = _compile(stmt)

        assert "websearch_to_tsquery" not in sql
        assert "WHERE" not in sql
        assert "ORDER BY vacancies.published_at DESC NULLS LAST" in sql
        assert "CAST(NULL AS FLOAT)" in sql
        # Без query параметров под rank не занято (он теперь CAST(...), не
        # bind), поэтому limit/offset — первые два именованных параметра.
        assert params["param_1"] == 20  # limit
        assert params["param_2"] == 0  # offset

    def test_none_query_behaves_like_empty(self) -> None:
        stmt = _search_statement(
            None,
            area_id=None,
            salary_min_rub=None,
            experience_ids=None,
            remote_only=False,
            published_within_days=None,
            limit=20,
            offset=0,
        )
        sql, _ = _compile(stmt)
        assert "websearch_to_tsquery" not in sql

    def test_text_query_ranks_and_filters_by_tsquery(self) -> None:
        stmt = _search_statement(
            "python  разработчик",
            area_id=None,
            salary_min_rub=None,
            experience_ids=None,
            remote_only=False,
            published_within_days=None,
            limit=20,
            offset=0,
        )
        sql, params = _compile(stmt)

        assert "websearch_to_tsquery(%(websearch_to_tsquery_1)s, %(websearch_to_tsquery_2)s)" in sql
        assert "vacancies.search_vector @@ websearch_to_tsquery" in sql
        assert "ORDER BY ts_rank_cd" in sql
        # первый аргумент websearch_to_tsquery — язык словаря, второй — сам запрос
        assert params["websearch_to_tsquery_1"] == "russian"
        assert params["websearch_to_tsquery_2"] == "python  разработчик"

    def test_all_filters_combine_with_and(self) -> None:
        stmt = _search_statement(
            "python",
            area_id=1,
            salary_min_rub=100_000,
            experience_ids=["between1And3", "between3And6"],
            remote_only=True,
            published_within_days=30,
            limit=20,
            offset=0,
        )
        sql, params = _compile(stmt)

        assert "vacancies.area_id = %(area_id_1)s" in sql
        assert "vacancies.salary_from_rub >= %(salary_from_rub_1)s" in sql
        assert "vacancies.experience_id IN" in sql
        assert "vacancies.is_remote IS true" in sql
        assert "vacancies.published_at >= now() - %(now_1)s" in sql
        assert " AND " in sql  # фильтры соединены через AND, не OR

        assert params["area_id_1"] == 1
        assert params["salary_from_rub_1"] == 100_000
        assert params["experience_id_1"] == ["between1And3", "between3And6"]
        assert params["now_1"] == timedelta(days=30)

    def test_remote_only_false_does_not_add_filter(self) -> None:
        stmt = _search_statement(
            "",
            area_id=None,
            salary_min_rub=None,
            experience_ids=None,
            remote_only=False,
            published_within_days=None,
            limit=20,
            offset=0,
        )
        sql, _ = _compile(stmt)
        # is_remote всё равно есть в списке колонок (это поле VacancySummary) —
        # проверяем именно отсутствие фильтра по нему, а не колонки вообще.
        assert "is_remote IS true" not in sql

    def test_empty_experience_ids_does_not_add_filter(self) -> None:
        stmt = _search_statement(
            "",
            area_id=None,
            salary_min_rub=None,
            experience_ids=[],
            remote_only=False,
            published_within_days=None,
            limit=20,
            offset=0,
        )
        sql, _ = _compile(stmt)
        assert "experience_id" not in sql

    def test_limit_above_max_is_clamped(self) -> None:
        stmt = _search_statement(
            "",
            area_id=None,
            salary_min_rub=None,
            experience_ids=None,
            remote_only=False,
            published_within_days=None,
            limit=10_000,
            offset=0,
        )
        _, params = _compile(stmt)
        assert params["param_1"] == MAX_LIMIT

    def test_limit_below_one_is_clamped_to_one(self) -> None:
        stmt = _search_statement(
            "",
            area_id=None,
            salary_min_rub=None,
            experience_ids=None,
            remote_only=False,
            published_within_days=None,
            limit=0,
            offset=0,
        )
        _, params = _compile(stmt)
        assert params["param_1"] == 1

    def test_negative_offset_is_clamped_to_zero(self) -> None:
        stmt = _search_statement(
            "",
            area_id=None,
            salary_min_rub=None,
            experience_ids=None,
            remote_only=False,
            published_within_days=None,
            limit=20,
            offset=-5,
        )
        _, params = _compile(stmt)
        assert params["param_2"] == 0


class TestApplyMarketFilters:
    def _base(self) -> Select[Any]:
        return select(func.count(Vacancy.id)).select_from(Vacancy)

    def test_no_filters_produces_no_where(self) -> None:
        sql, _ = _compile(
            _apply_market_filters(
                self._base(), query=None, published_within_days=None, area_id=None
            )
        )
        assert "WHERE" not in sql

    def test_query_and_area_and_freshness_combine(self) -> None:
        stmt = _apply_market_filters(
            self._base(), query="backend", published_within_days=7, area_id=2
        )
        sql, params = _compile(stmt)

        assert "vacancies.search_vector @@ websearch_to_tsquery" in sql
        assert "vacancies.area_id = %(area_id_1)s" in sql
        assert "vacancies.published_at >= now() - %(now_1)s" in sql
        assert params["area_id_1"] == 2
        assert params["now_1"] == timedelta(days=7)

    def test_blank_query_is_treated_as_no_query(self) -> None:
        sql, _ = _compile(
            _apply_market_filters(
                self._base(), query="   ", published_within_days=None, area_id=None
            )
        )
        assert "websearch_to_tsquery" not in sql


class TestSkillStatsStatement:
    def test_groups_orders_and_limits(self) -> None:
        stmt = _skill_stats_statement(query=None, published_within_days=None, area_id=None, top_n=5)
        sql, params = _compile(stmt)

        assert "GROUP BY skills.id, skills.name, skills.display_name" in sql
        assert "ORDER BY count(vacancies.id) DESC" in sql
        assert (
            "percentile_cont(%(percentile_cont_1)s) WITHIN GROUP (ORDER BY vacancies.salary_from_rub)"
            in sql
        )
        assert "FILTER (WHERE vacancies.salary_from_rub IS NOT NULL)" in sql
        assert params["param_1"] == 5

    def test_top_n_above_max_is_clamped(self) -> None:
        stmt = _skill_stats_statement(
            query=None, published_within_days=None, area_id=None, top_n=9_999
        )
        _, params = _compile(stmt)
        assert params["param_1"] == MAX_LIMIT


# ------------------------------------------------------------- сессия-заглушка --


class _FakeResult:
    """Заглушка результата session.execute() — отдаёт заранее заданные данные."""

    def __init__(
        self,
        *,
        mappings: list[dict[str, Any]] | None = None,
        rows: list[Any] | None = None,
        scalar: Any = None,
        row: Any = None,
    ) -> None:
        self._mappings = mappings
        self._rows = rows
        self._scalar = scalar
        self._row = row

    def mappings(self) -> _FakeResult:
        return self

    def all(self) -> list[Any]:
        if self._mappings is not None:
            return self._mappings
        assert self._rows is not None
        return self._rows

    def scalar_one(self) -> Any:
        return self._scalar

    def scalar_one_or_none(self) -> Any:
        return self._scalar

    def one(self) -> Any:
        return self._row


class _FakeSession:
    """Сессия-заглушка: отдаёт результаты по очереди — по одному на каждый execute().

    Каждый переданный statement дополнительно компилируется под postgres —
    это ловит синтаксические ошибки SQL и в тех тестах, которые сами по себе
    не разбирают текст запроса, а проверяют только маппинг результата.
    """

    def __init__(self, *results: _FakeResult) -> None:
        self._results = list(results)
        self.captured_statements: list[Any] = []

    def execute(self, stmt: Any) -> _FakeResult:
        stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": False})
        self.captured_statements.append(stmt)
        return self._results.pop(0)


class TestSearchVacanciesMapping:
    def test_maps_rows_into_vacancy_summary(self) -> None:
        row = {
            "id": 1,
            "name": "Backend-разработчик",
            "employer_name": "Acme",
            "area_name": "Москва",
            "salary_from_rub": 150_000,
            "salary_to_rub": None,
            "salary_currency": "RUR",
            "experience_name": "От 1 года до 3 лет",
            "is_remote": True,
            "published_at": datetime(2026, 8, 1, tzinfo=UTC),
            "alternate_url": "https://hh.ru/vacancy/1",
            "description": "Пишем на Python",
            "rank": 0.42,
        }
        session = _FakeSession(_FakeResult(mappings=[row]))

        results = search_vacancies(session, "python", limit=9999)  # type: ignore[arg-type]

        assert results == [
            VacancySummary(
                id=1,
                name="Backend-разработчик",
                employer_name="Acme",
                area_name="Москва",
                salary_from_rub=150_000,
                salary_to_rub=None,
                salary_currency="RUR",
                experience_name="От 1 года до 3 лет",
                is_remote=True,
                published_at=row["published_at"],
                alternate_url="https://hh.ru/vacancy/1",
                description="Пишем на Python",
                rank=0.42,
            )
        ]
        assert results[0].to_dict()["published_at"] == row["published_at"].isoformat()

        # limit из вызова (9999) должен был приехать в execute() уже клампнутым.
        # При непустом query первые именованные параметры уходят под
        # websearch_to_tsquery, поэтому LIMIT — это param_1, а не param_2.
        compiled = session.captured_statements[0].compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": False}
        )
        assert compiled.params["param_1"] == MAX_LIMIT

    def test_empty_result_is_empty_list(self) -> None:
        session = _FakeSession(_FakeResult(mappings=[]))
        assert search_vacancies(session, None) == []  # type: ignore[arg-type]


class TestGetVacancyMapping:
    def test_returns_none_when_not_found(self) -> None:
        session = _FakeSession(_FakeResult(scalar=None))
        assert get_vacancy(session, 12345) is None  # type: ignore[arg-type]

    def test_maps_loaded_orm_object_into_detail(self) -> None:
        # Транзиентные ORM-объекты, ни к какой сессии не привязанные — то, что
        # вернул бы selectinload(employer, skills), не поднимая базу.
        employer = Employer(
            id=5, name="Acme", alternate_url="https://hh.ru/employer/5", trusted=True
        )
        vacancy = Vacancy(
            id=1,
            name="Backend-разработчик",
            employer=employer,
            area_id=1,
            area_name="Москва",
            salary_from=150_000,
            salary_to=None,
            salary_currency="RUR",
            salary_gross=False,
            salary_from_rub=150_000,
            salary_to_rub=None,
            experience_id="between1And3",
            experience_name="От 1 года до 3 лет",
            employment_id="full",
            schedule_id="remote",
            is_remote=True,
            professional_roles=None,
            description="Пишем на Python",
            alternate_url="https://hh.ru/vacancy/1",
            published_at=None,
            created_at=None,
            archived=False,
        )
        vacancy.skills = [
            Skill(id=1, name="postgresql", display_name="PostgreSQL"),
            Skill(id=2, name="python", display_name="Python"),
        ]
        session = _FakeSession(_FakeResult(scalar=vacancy))

        detail = get_vacancy(session, 1)  # type: ignore[arg-type]

        assert detail is not None
        assert detail.employer == EmployerInfo(
            id=5, name="Acme", alternate_url="https://hh.ru/employer/5", trusted=True
        )
        assert detail.skills == ["postgresql", "python"]  # отсортированы
        assert detail.salary_from_rub == 150_000


class TestSkillStatsMapping:
    def test_returns_empty_list_without_second_query_when_no_matches(self) -> None:
        session = _FakeSession(_FakeResult(scalar=0))

        result = skill_stats(session)  # type: ignore[arg-type]

        assert result == []
        assert len(session.captured_statements) == 1  # запрос за списком навыков не выполнялся


class TestMarketOverviewMapping:
    def test_compiles_all_three_queries_and_maps_result(self) -> None:
        """market_overview собирает три отдельных запроса (общие агрегаты,
        разбивка по опыту, топ работодателей) — здесь же неявно проверяется,
        что все три компилируются (percentile_cont-тройка, cast, join'ы —
        самая насыщенная SQL-конструкция модуля)."""
        agg_row = SimpleNamespace(
            total=10, with_salary=5, p25=100_000, p50=120_000, p75=150_000, remote_share=0.5
        )
        exp_rows = [
            SimpleNamespace(experience_id="between1And3", n=7),
            SimpleNamespace(experience_id="noExperience", n=3),
        ]
        emp_rows = [SimpleNamespace(name="Acme", n=4), SimpleNamespace(name="Beta", n=2)]
        session = _FakeSession(
            _FakeResult(row=agg_row),
            _FakeResult(rows=exp_rows),
            _FakeResult(rows=emp_rows),
        )

        overview = market_overview(  # type: ignore[arg-type]
            session, query="python", published_within_days=30, area_id=1
        )

        assert overview == MarketOverview(
            total=10,
            with_salary=5,
            salary_p25=100_000,
            salary_p50=120_000,
            salary_p75=150_000,
            remote_share=0.5,
            by_experience=[("between1And3", 7), ("noExperience", 3)],
            top_employers=[("Acme", 4), ("Beta", 2)],
        )
        assert len(session.captured_statements) == 3

    def test_missing_percentiles_and_remote_share_become_none_and_zero(self) -> None:
        """Когда подходящих вакансий нет вовсе, percentile_cont и avg отдают NULL —
        это не должно всплыть наружу как исключение при int()/float()."""
        agg_row = SimpleNamespace(
            total=0, with_salary=0, p25=None, p50=None, p75=None, remote_share=None
        )
        session = _FakeSession(_FakeResult(row=agg_row), _FakeResult(rows=[]), _FakeResult(rows=[]))

        overview = market_overview(session)  # type: ignore[arg-type]

        assert overview.salary_p25 is None
        assert overview.remote_share == 0.0


class TestDbStatusMapping:
    def test_maps_counts_and_bounds(self) -> None:
        published_from = datetime(2026, 1, 1, tzinfo=UTC)
        published_to = datetime(2026, 8, 1, tzinfo=UTC)
        last_fetched = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)
        session = _FakeSession(
            _FakeResult(scalar=120),  # vacancies_total
            _FakeResult(scalar=45),  # employers_total
            _FakeResult(scalar=300),  # skills_total
            _FakeResult(scalar=980),  # chunks_total
            _FakeResult(scalar=970),  # chunks_embedded
            _FakeResult(scalar="paraphrase-multilingual-MiniLM-L12-v2"),  # model_name
            _FakeResult(row=(published_from, published_to, last_fetched)),  # bounds
        )

        status = db_status(session)  # type: ignore[arg-type]

        assert status == DbStatus(
            vacancies_total=120,
            employers_total=45,
            skills_total=300,
            chunks_total=980,
            chunks_embedded=970,
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
            published_from=published_from,
            published_to=published_to,
            last_fetched_at=last_fetched,
        )
        assert status.to_dict()["published_from"] == published_from.isoformat()

    def test_model_name_is_none_when_there_are_no_chunks(self) -> None:
        session = _FakeSession(
            _FakeResult(scalar=0),
            _FakeResult(scalar=0),
            _FakeResult(scalar=0),
            _FakeResult(scalar=0),
            _FakeResult(scalar=0),
            _FakeResult(scalar=None),  # group by на пустой таблице — ни одной группы
            _FakeResult(row=(None, None, None)),
        )

        status = db_status(session)  # type: ignore[arg-type]

        assert status.model_name is None
