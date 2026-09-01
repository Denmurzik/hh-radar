"""Слой запросов к базе.

Все функции принимают уже открытую ``Session`` первым аргументом и возвращают
dataclass'ы или примитивы — никогда ORM-объекты. Причина: MCP-слой открывает
короткоживущую сессию на каждый вызов инструмента (``session_scope()``) и
закрывает её сразу после запроса, а ORM-объект после закрытия сессии либо
детачится, либо роняет ``DetachedInstanceError`` при обращении к непрогруженным
атрибутам. Явные dataclass'ы этой проблемы не знают и сериализуются в JSON
напрямую через ``to_dict()``.

Здесь же сосредоточена вся SQL-логика проекта: построение выражений вынесено
в приватные функции (``_search_statement``, ``_apply_market_filters`` и т.д.),
чтобы их можно было компилировать и проверять в тестах без поднятой базы.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import ColumnElement, Float, Select, case, cast, func, null, select
from sqlalchemy.orm import Session, selectinload

from hh_radar.db.models import Employer, Skill, Vacancy, VacancyChunk, vacancy_skills

#: Верхняя граница для любого параметра limit/top_n в этом модуле. MCP-слой
#: полагается на неё как на последний рубеж защиты от выгрузки всей базы в
#: контекст модели, но клампает здесь, а не только снаружи — функции модуля
#: должны быть безопасны и при прямом вызове, в обход MCP.
MAX_LIMIT = 50

#: Порядок требуемого опыта в терминах hh.ru — от отсутствия опыта до 6+ лет.
#: Используется в profile.match_vacancy для сравнения "выше/ниже" требования.
EXPERIENCE_ORDER: dict[str, int] = {
    "noExperience": 0,
    "between1And3": 1,
    "between3And6": 2,
    "moreThan6": 3,
}


def _iso(value: datetime | None) -> str | None:
    """Сериализация datetime в ISO-строку для to_dict(); None остаётся None."""
    return value.isoformat() if value is not None else None


def _clamp_limit(limit: int) -> int:
    """Ограничивает limit диапазоном [1, MAX_LIMIT]."""
    return max(1, min(limit, MAX_LIMIT))


@dataclass(frozen=True, slots=True)
class VacancySummary:
    """Короткая карточка вакансии — то, что возвращает поиск."""

    id: int
    name: str
    employer_name: str | None
    area_name: str | None
    salary_from_rub: int | None
    salary_to_rub: int | None
    salary_currency: str | None
    experience_name: str | None
    is_remote: bool
    published_at: datetime | None
    alternate_url: str | None
    description: str | None
    rank: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "employer_name": self.employer_name,
            "area_name": self.area_name,
            "salary_from_rub": self.salary_from_rub,
            "salary_to_rub": self.salary_to_rub,
            "salary_currency": self.salary_currency,
            "experience_name": self.experience_name,
            "is_remote": self.is_remote,
            "published_at": _iso(self.published_at),
            "alternate_url": self.alternate_url,
            "description": self.description,
            "rank": self.rank,
        }


@dataclass(frozen=True, slots=True)
class EmployerInfo:
    """Работодатель внутри полной карточки вакансии."""

    id: int
    name: str
    alternate_url: str | None
    trusted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "alternate_url": self.alternate_url,
            "trusted": self.trusted,
        }


@dataclass(frozen=True, slots=True)
class VacancyDetail:
    """Полная карточка вакансии: все поля, работодатель, навыки.

    ``skills`` — нормализованные имена (``Skill.name``), а не то, как их
    написал работодатель: именно в таком виде их сравнивает
    ``hh_radar.profile.match_vacancy`` с профилем кандидата.
    """

    id: int
    name: str
    employer: EmployerInfo | None
    area_id: int | None
    area_name: str | None
    salary_from: int | None
    salary_to: int | None
    salary_currency: str | None
    salary_gross: bool | None
    salary_from_rub: int | None
    salary_to_rub: int | None
    experience_id: str | None
    experience_name: str | None
    employment_id: str | None
    schedule_id: str | None
    is_remote: bool
    professional_roles: list[dict[str, Any]] | None
    description: str | None
    alternate_url: str | None
    published_at: datetime | None
    created_at: datetime | None
    archived: bool
    skills: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "employer": self.employer.to_dict() if self.employer is not None else None,
            "area_id": self.area_id,
            "area_name": self.area_name,
            "salary_from": self.salary_from,
            "salary_to": self.salary_to,
            "salary_currency": self.salary_currency,
            "salary_gross": self.salary_gross,
            "salary_from_rub": self.salary_from_rub,
            "salary_to_rub": self.salary_to_rub,
            "experience_id": self.experience_id,
            "experience_name": self.experience_name,
            "employment_id": self.employment_id,
            "schedule_id": self.schedule_id,
            "is_remote": self.is_remote,
            "professional_roles": self.professional_roles,
            "description": self.description,
            "alternate_url": self.alternate_url,
            "published_at": _iso(self.published_at),
            "created_at": _iso(self.created_at),
            "archived": self.archived,
            "skills": self.skills,
        }


@dataclass(frozen=True, slots=True)
class SkillStat:
    """Строка агрегации «сколько вакансий требуют этот навык»."""

    name: str
    display_name: str
    vacancy_count: int
    share: float
    median_salary_from_rub: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "vacancy_count": self.vacancy_count,
            "share": self.share,
            "median_salary_from_rub": self.median_salary_from_rub,
        }


@dataclass(frozen=True, slots=True)
class MarketOverview:
    """Срез рынка по фильтру: зарплаты, удалёнка, опыт, работодатели."""

    total: int
    with_salary: int
    salary_p25: int | None
    salary_p50: int | None
    salary_p75: int | None
    remote_share: float
    by_experience: list[tuple[str, int]]
    top_employers: list[tuple[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "with_salary": self.with_salary,
            "salary_p25": self.salary_p25,
            "salary_p50": self.salary_p50,
            "salary_p75": self.salary_p75,
            "remote_share": self.remote_share,
            "by_experience": [
                {"experience_id": exp_id, "count": count} for exp_id, count in self.by_experience
            ],
            "top_employers": [
                {"employer_name": name, "count": count} for name, count in self.top_employers
            ],
        }


@dataclass(frozen=True, slots=True)
class DbStatus:
    """Что лежит в базе — чтобы агент знал границы своих данных."""

    vacancies_total: int
    employers_total: int
    skills_total: int
    chunks_total: int
    chunks_embedded: int
    #: Какой моделью эмбеддингов чаще всего проиндексированы чанки — None,
    #: если чанков нет вовсе. Одна база в теории может содержать чанки
    #: разных моделей (например, после смены EMBEDDING_MODEL), поэтому это
    #: самое частое значение, а не гарантированно единственное.
    model_name: str | None
    published_from: datetime | None
    published_to: datetime | None
    last_fetched_at: datetime | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "vacancies_total": self.vacancies_total,
            "employers_total": self.employers_total,
            "skills_total": self.skills_total,
            "chunks_total": self.chunks_total,
            "chunks_embedded": self.chunks_embedded,
            "model_name": self.model_name,
            "published_from": _iso(self.published_from),
            "published_to": _iso(self.published_to),
            "last_fetched_at": _iso(self.last_fetched_at),
        }


def _apply_market_filters(
    stmt: Select[Any],
    *,
    query: str | None,
    published_within_days: int | None,
    area_id: int | None,
) -> Select[Any]:
    """Общие фильтры среза рынка: текстовый запрос, регион, свежесть публикации.

    Переиспользуется в skill_stats и market_overview — оба считают агрегаты
    по одному и тому же подмножеству вакансий, и рассинхронизация фильтров
    между ними была бы тихой, но неприятной ошибкой (числа бы просто не
    сходились друг с другом).
    """
    text_query = (query or "").strip()
    if text_query:
        tsquery = func.websearch_to_tsquery("russian", text_query)
        stmt = stmt.where(Vacancy.search_vector.bool_op("@@")(tsquery))
    if area_id is not None:
        stmt = stmt.where(Vacancy.area_id == area_id)
    if published_within_days is not None:
        stmt = stmt.where(
            Vacancy.published_at >= func.now() - timedelta(days=published_within_days)
        )
    return stmt


def _search_statement(
    query: str | None,
    *,
    area_id: int | None,
    salary_min_rub: int | None,
    experience_ids: Sequence[str] | None,
    remote_only: bool,
    published_within_days: int | None,
    limit: int,
    offset: int,
) -> Select[Any]:
    """Строит запрос полнотекстового поиска. Вынесен отдельно ради тестируемости:

    тесты компилируют результат этой функции и проверяют попадание фильтров,
    ORDER BY и LIMIT в SQL — без обращения к живой базе.
    """
    text_query = (query or "").strip()
    # Явная аннотация нужна mypy: в двух ветках ниже присваиваются разные
    # конкретные подклассы ColumnElement (Function и BindParameter), и без
    # общего типа он требует их дословного совпадения.
    rank_expr: ColumnElement[Any]
    if text_query:
        tsquery = func.websearch_to_tsquery("russian", text_query)
        rank_expr = func.ts_rank_cd(Vacancy.search_vector, tsquery)
    else:
        tsquery = None
        # literal(None, type_=Float) рендерится как бестиповый NULL-bind —
        # Postgres выводит для него text (OID 25), и psycopg падает при
        # разборе результата ("Unknown PG numeric type: 25"). CAST(NULL AS
        # FLOAT) фиксирует тип на стороне SQL, а не только в Python.
        rank_expr = cast(null(), Float)

    stmt = (
        select(
            Vacancy.id,
            Vacancy.name,
            Employer.name.label("employer_name"),
            Vacancy.area_name,
            Vacancy.salary_from_rub,
            Vacancy.salary_to_rub,
            Vacancy.salary_currency,
            Vacancy.experience_name,
            Vacancy.is_remote,
            Vacancy.published_at,
            Vacancy.alternate_url,
            Vacancy.description,
            rank_expr.label("rank"),
        )
        .select_from(Vacancy)
        .outerjoin(Employer, Vacancy.employer_id == Employer.id)
    )

    if tsquery is not None:
        stmt = stmt.where(Vacancy.search_vector.bool_op("@@")(tsquery))
    if area_id is not None:
        stmt = stmt.where(Vacancy.area_id == area_id)
    if salary_min_rub is not None:
        stmt = stmt.where(Vacancy.salary_from_rub >= salary_min_rub)
    if experience_ids:
        stmt = stmt.where(Vacancy.experience_id.in_(experience_ids))
    if remote_only:
        stmt = stmt.where(Vacancy.is_remote.is_(True))
    if published_within_days is not None:
        stmt = stmt.where(
            Vacancy.published_at >= func.now() - timedelta(days=published_within_days)
        )

    order_by = []
    if tsquery is not None:
        order_by.append(rank_expr.desc())
    order_by.append(Vacancy.published_at.desc().nulls_last())
    stmt = stmt.order_by(*order_by)

    return stmt.limit(_clamp_limit(limit)).offset(max(0, offset))


def search_vacancies(
    session: Session,
    query: str | None,
    *,
    area_id: int | None = None,
    salary_min_rub: int | None = None,
    experience_ids: Sequence[str] | None = None,
    remote_only: bool = False,
    published_within_days: int | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[VacancySummary]:
    """Полнотекстовый поиск вакансий по названию и описанию.

    Пустой или отсутствующий ``query`` — не ошибка: возвращаются вакансии,
    подходящие под остальные фильтры, отсортированные по дате публикации.
    При непустом query ранжирование ведётся по ``ts_rank_cd``, при равенстве
    ранга — по дате (вторичная сортировка).
    """
    stmt = _search_statement(
        query,
        area_id=area_id,
        salary_min_rub=salary_min_rub,
        experience_ids=experience_ids,
        remote_only=remote_only,
        published_within_days=published_within_days,
        limit=limit,
        offset=offset,
    )
    rows = session.execute(stmt).mappings().all()
    return [
        VacancySummary(
            id=row["id"],
            name=row["name"],
            employer_name=row["employer_name"],
            area_name=row["area_name"],
            salary_from_rub=row["salary_from_rub"],
            salary_to_rub=row["salary_to_rub"],
            salary_currency=row["salary_currency"],
            experience_name=row["experience_name"],
            is_remote=row["is_remote"],
            published_at=row["published_at"],
            alternate_url=row["alternate_url"],
            description=row["description"],
            rank=row["rank"],
        )
        for row in rows
    ]


def _vacancy_to_detail(vacancy: Vacancy) -> VacancyDetail:
    """Собирает VacancyDetail из уже загруженного ORM-объекта.

    Ожидает, что ``employer`` и ``skills`` уже загружены (через selectinload) —
    функция сама к базе не обращается, поэтому она тестируется на вручную
    собранном, ни к какой сессии не привязанном объекте Vacancy.
    """
    employer = None
    if vacancy.employer is not None:
        employer = EmployerInfo(
            id=vacancy.employer.id,
            name=vacancy.employer.name,
            alternate_url=vacancy.employer.alternate_url,
            trusted=vacancy.employer.trusted,
        )
    return VacancyDetail(
        id=vacancy.id,
        name=vacancy.name,
        employer=employer,
        area_id=vacancy.area_id,
        area_name=vacancy.area_name,
        salary_from=vacancy.salary_from,
        salary_to=vacancy.salary_to,
        salary_currency=vacancy.salary_currency,
        salary_gross=vacancy.salary_gross,
        salary_from_rub=vacancy.salary_from_rub,
        salary_to_rub=vacancy.salary_to_rub,
        experience_id=vacancy.experience_id,
        experience_name=vacancy.experience_name,
        employment_id=vacancy.employment_id,
        schedule_id=vacancy.schedule_id,
        is_remote=vacancy.is_remote,
        professional_roles=vacancy.professional_roles,
        description=vacancy.description,
        alternate_url=vacancy.alternate_url,
        published_at=vacancy.published_at,
        created_at=vacancy.created_at,
        archived=vacancy.archived,
        skills=sorted(skill.name for skill in vacancy.skills),
    )


def get_vacancy(session: Session, vacancy_id: int) -> VacancyDetail | None:
    """Полная карточка вакансии по id, либо None, если такой вакансии нет."""
    stmt = (
        select(Vacancy)
        .where(Vacancy.id == vacancy_id)
        .options(selectinload(Vacancy.employer), selectinload(Vacancy.skills))
    )
    vacancy = session.execute(stmt).scalar_one_or_none()
    if vacancy is None:
        return None
    return _vacancy_to_detail(vacancy)


def _skill_stats_statement(
    *,
    query: str | None,
    published_within_days: int | None,
    area_id: int | None,
    top_n: int,
) -> Select[Any]:
    """Запрос агрегации по навыкам. Вынесен отдельно для компиляции в тестах."""
    stmt = (
        select(
            Skill.name,
            Skill.display_name,
            func.count(Vacancy.id).label("vacancy_count"),
            func.percentile_cont(0.5)
            .within_group(Vacancy.salary_from_rub)
            .filter(Vacancy.salary_from_rub.is_not(None))
            .label("median_salary"),
        )
        .select_from(Vacancy)
        .join(vacancy_skills, vacancy_skills.c.vacancy_id == Vacancy.id)
        .join(Skill, Skill.id == vacancy_skills.c.skill_id)
        .group_by(Skill.id, Skill.name, Skill.display_name)
        .order_by(func.count(Vacancy.id).desc())
        .limit(_clamp_limit(top_n))
    )
    return _apply_market_filters(
        stmt, query=query, published_within_days=published_within_days, area_id=area_id
    )


def skill_stats(
    session: Session,
    *,
    query: str | None = None,
    published_within_days: int | None = None,
    area_id: int | None = None,
    top_n: int = 30,
) -> list[SkillStat]:
    """Топ навыков среди подходящих под фильтр вакансий.

    ``share`` — доля вакансий с этим навыком от общего числа подходящих под
    фильтр (не от числа вакансий с навыками вообще). ``median_salary_from_rub``
    считается только по вакансиям, где зарплата указана — вакансии без
    зарплаты в медиану не попадают, а не трактуются как ноль.
    """
    total_stmt = _apply_market_filters(
        select(func.count(Vacancy.id)).select_from(Vacancy),
        query=query,
        published_within_days=published_within_days,
        area_id=area_id,
    )
    total = session.execute(total_stmt).scalar_one()
    if not total:
        return []

    stmt = _skill_stats_statement(
        query=query, published_within_days=published_within_days, area_id=area_id, top_n=top_n
    )
    rows = session.execute(stmt).all()
    return [
        SkillStat(
            name=row.name,
            display_name=row.display_name,
            vacancy_count=row.vacancy_count,
            share=row.vacancy_count / total,
            median_salary_from_rub=(
                int(row.median_salary) if row.median_salary is not None else None
            ),
        )
        for row in rows
    ]


def market_overview(
    session: Session,
    *,
    query: str | None = None,
    published_within_days: int | None = None,
    area_id: int | None = None,
) -> MarketOverview:
    """Общий срез рынка по фильтру: зарплаты, удалёнка, опыт, работодатели."""
    agg_stmt = _apply_market_filters(
        select(
            func.count(Vacancy.id).label("total"),
            func.count(Vacancy.salary_from_rub).label("with_salary"),
            func.percentile_cont(0.25)
            .within_group(Vacancy.salary_from_rub)
            .filter(Vacancy.salary_from_rub.is_not(None))
            .label("p25"),
            func.percentile_cont(0.5)
            .within_group(Vacancy.salary_from_rub)
            .filter(Vacancy.salary_from_rub.is_not(None))
            .label("p50"),
            func.percentile_cont(0.75)
            .within_group(Vacancy.salary_from_rub)
            .filter(Vacancy.salary_from_rub.is_not(None))
            .label("p75"),
            # CAST(boolean AS float) Postgres не умеет напрямую ("cannot cast
            # type boolean to double precision") — падает только на живой
            # базе, компиляция SQL этого не ловит. CASE — рабочий обходной путь.
            func.avg(case((Vacancy.is_remote, 1.0), else_=0.0)).label("remote_share"),
        ).select_from(Vacancy),
        query=query,
        published_within_days=published_within_days,
        area_id=area_id,
    )
    row = session.execute(agg_stmt).one()

    experience_stmt = _apply_market_filters(
        select(Vacancy.experience_id, func.count(Vacancy.id).label("n"))
        .select_from(Vacancy)
        .where(Vacancy.experience_id.is_not(None))
        .group_by(Vacancy.experience_id)
        .order_by(func.count(Vacancy.id).desc()),
        query=query,
        published_within_days=published_within_days,
        area_id=area_id,
    )
    by_experience = [(r.experience_id, r.n) for r in session.execute(experience_stmt).all()]

    employer_stmt = _apply_market_filters(
        select(Employer.name, func.count(Vacancy.id).label("n"))
        .select_from(Vacancy)
        .join(Employer, Vacancy.employer_id == Employer.id)
        .group_by(Employer.id, Employer.name)
        .order_by(func.count(Vacancy.id).desc())
        .limit(10),
        query=query,
        published_within_days=published_within_days,
        area_id=area_id,
    )
    top_employers = [(r.name, r.n) for r in session.execute(employer_stmt).all()]

    return MarketOverview(
        total=row.total,
        with_salary=row.with_salary,
        salary_p25=int(row.p25) if row.p25 is not None else None,
        salary_p50=int(row.p50) if row.p50 is not None else None,
        salary_p75=int(row.p75) if row.p75 is not None else None,
        remote_share=float(row.remote_share) if row.remote_share is not None else 0.0,
        by_experience=by_experience,
        top_employers=top_employers,
    )


def db_status(session: Session) -> DbStatus:
    """Сколько чего лежит в базе и за какой период — границы данных для агента."""
    vacancies_total = session.execute(select(func.count(Vacancy.id))).scalar_one()
    employers_total = session.execute(select(func.count(Employer.id))).scalar_one()
    skills_total = session.execute(select(func.count(Skill.id))).scalar_one()
    chunks_total = session.execute(select(func.count(VacancyChunk.id))).scalar_one()
    chunks_embedded = session.execute(
        select(func.count(VacancyChunk.id)).where(VacancyChunk.embedding.is_not(None))
    ).scalar_one()
    model_name = session.execute(
        select(VacancyChunk.model_name)
        .group_by(VacancyChunk.model_name)
        .order_by(func.count(VacancyChunk.id).desc())
        .limit(1)
    ).scalar_one_or_none()
    bounds = session.execute(
        select(
            func.min(Vacancy.published_at),
            func.max(Vacancy.published_at),
            func.max(Vacancy.fetched_at),
        )
    ).one()
    return DbStatus(
        vacancies_total=vacancies_total,
        employers_total=employers_total,
        skills_total=skills_total,
        chunks_total=chunks_total,
        chunks_embedded=chunks_embedded,
        model_name=model_name,
        published_from=bounds[0],
        published_to=bounds[1],
        last_fetched_at=bounds[2],
    )
