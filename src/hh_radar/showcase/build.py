"""Сборка витрины из данных базы.

Витрина существует по одной причине: проверяющий не станет клонировать
репозиторий и поднимать docker compose. Ему нужна ссылка, которая открывается
за десять секунд и показывает, что за кодом стоят реальные данные.

Все цифры считаются здесь и сейчас из базы. Рядом с HTML кладётся ``data.json``
с теми же числами — чтобы утверждение «ничего не вписано руками» можно было
проверить, а не принять на веру.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hh_radar.db.models import Vacancy
from hh_radar.db.queries import (
    MarketOverview,
    SkillStat,
    db_status,
    market_overview,
    skill_stats,
)
from hh_radar.showcase.render import BarRow, TimePoint, experience_label, render_page
from hh_radar.text import format_count, format_money, plural_ru

logger = logging.getLogger(__name__)

TOP_SKILLS = 15
TOP_EMPLOYERS = 10


@dataclass(frozen=True, slots=True)
class ShowcaseData:
    """Всё, что нужно странице. Отделено от отрисовки, чтобы это можно было
    сериализовать в JSON и проверить отдельно от вёрстки."""

    generated_at: datetime
    stats: list[tuple[str, str, str]]
    skills: list[BarRow]
    salaries: list[BarRow]
    timeline: list[TimePoint]
    employers: list[tuple[str, int]]
    examples: list[tuple[str, str]]
    period: tuple[datetime | None, datetime | None]
    payload: dict[str, object]
    vacancies_total: int


def build_showcase(session: Session, out_dir: Path) -> list[Path]:
    """Собрать ``index.html`` и ``data.json`` в указанном каталоге."""
    data = collect(session)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = out_dir / "index.html"
    index.write_text(
        render_page(
            generated_at=data.generated_at,
            stats=data.stats,
            skills=data.skills,
            salaries=data.salaries,
            timeline=data.timeline,
            employers=data.employers,
            examples=data.examples,
            period=data.period,
            payload=data.payload,
            vacancies_total=data.vacancies_total,
        ),
        encoding="utf-8",
    )

    payload = out_dir / "data.json"
    payload.write_text(
        json.dumps(data.payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # .nojekyll — иначе GitHub Pages прогоняет каталог через Jekyll и
    # выбрасывает всё, что начинается с подчёркивания.
    nojekyll = out_dir / ".nojekyll"
    nojekyll.write_text("", encoding="utf-8")

    return [index, payload, nojekyll]


def collect(session: Session) -> ShowcaseData:
    """Посчитать все числа витрины одним заходом."""
    status = db_status(session)
    overview = market_overview(session)
    skills = skill_stats(session, top_n=TOP_SKILLS)
    weekly = _weekly_counts(session)

    stats = [
        (
            "вакансий в базе",
            format_count(status.vacancies_total),
            f"работодателей: {format_count(status.employers_total)}",
        ),
        (
            "медиана зарплаты",
            format_money(overview.salary_p50),
            f"вилку указали {_percent(overview.with_salary, overview.total)} вакансий",
        ),
        (
            "удалённых вакансий",
            f"{overview.remote_share * 100:.0f}%",
            "по полям work_format и schedule",
        ),
        (
            "уникальных навыков",
            format_count(status.skills_total),
            f"векторов для поиска: {format_count(status.chunks_embedded)}",
        ),
    ]

    skill_rows = [
        BarRow(
            label=stat.display_name,
            value=stat.vacancy_count,
            display=f"{stat.share * 100:.0f}%",
            note=(
                f"{format_count(stat.vacancy_count)} "
                + plural_ru(stat.vacancy_count, "вакансия", "вакансии", "вакансий")
                + (
                    f", медиана {format_money(stat.median_salary_from_rub)}"
                    if stat.median_salary_from_rub
                    else ""
                )
            ),
        )
        for stat in skills
    ]

    salary_rows = _salary_by_experience(session)

    examples = _examples(overview, skills)

    payload: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "database": status.to_dict(),
        "market": overview.to_dict(),
        "top_skills": [stat.to_dict() for stat in skills],
        "salary_by_experience": [
            {"experience": row.label, "median_salary_from_rub": row.value} for row in salary_rows
        ],
        "weekly_published": [
            {"week_start": point.moment.date().isoformat(), "count": point.count}
            for point in weekly
        ],
    }

    return ShowcaseData(
        generated_at=datetime.now(UTC),
        stats=stats,
        skills=skill_rows,
        salaries=salary_rows,
        timeline=weekly,
        employers=overview.top_employers[:TOP_EMPLOYERS],
        examples=examples,
        period=(status.published_from, status.published_to),
        payload=payload,
        vacancies_total=status.vacancies_total,
    )


# ------------------------------------------------------------- аггрегаты ---


def _weekly_counts(session: Session) -> list[TimePoint]:
    """Публикации по неделям.

    ``date_trunc('week', ...)`` считает на стороне базы: тянуть все даты
    в питон ради группировки — лишний трафик и лишняя память.
    """
    bucket = func.date_trunc("week", Vacancy.published_at).label("week")
    rows = session.execute(
        select(bucket, func.count())
        .where(Vacancy.published_at.is_not(None))
        .group_by(bucket)
        .order_by(bucket)
    ).all()
    return [TimePoint(moment=row[0], count=row[1]) for row in rows if row[0] is not None]


def _salary_by_experience(session: Session) -> list[BarRow]:
    """Медиана нижней границы вилки по уровню опыта.

    Медиана, а не среднее: одна вакансия «от 900 000 ₽» перекашивает среднее
    так, что цифра перестаёт что-либо значить.
    """
    median = func.percentile_cont(0.5).within_group(Vacancy.salary_from_rub.asc())
    rows = session.execute(
        select(Vacancy.experience_id, median, func.count())
        .where(Vacancy.salary_from_rub.is_not(None))
        .group_by(Vacancy.experience_id)
        .order_by(median.desc())
    ).all()

    return [
        BarRow(
            label=experience_label(row[0]),
            value=float(row[1] or 0),
            display=format_money(int(row[1])) if row[1] else "—",
            note=(
                f"{format_count(row[2])} "
                + plural_ru(row[2], "вакансия", "вакансии", "вакансий")
                + " с указанной вилкой"
            ),
        )
        for row in rows
        if row[1]
    ]


def _examples(overview: MarketOverview, skills: list[SkillStat]) -> list[tuple[str, str]]:
    """Живые примеры вопросов к агенту с ответами из тех же данных.

    Ответы собираются из посчитанных чисел, а не пишутся текстом: если база
    поменяется, поменяются и они.
    """
    examples: list[tuple[str, str]] = []

    if skills:
        top = skills[0]
        examples.append(
            (
                "Какие навыки чаще всего требуют в вакансиях по AI и автоматизации?",
                f"Чаще всего — {top.display_name}: {top.share * 100:.0f}% вакансий "
                f"({format_count(top.vacancy_count)} шт.). "
                + (
                    "Дальше: " + ", ".join(stat.display_name for stat in skills[1:5]) + "."
                    if len(skills) > 1
                    else ""
                ),
            )
        )

    p25 = overview.salary_p25
    p75 = overview.salary_p75
    if p25 and p75:
        examples.append(
            (
                "На какую вилку рассчитывать?",
                f"Средние 50% вакансий с указанной зарплатой лежат между "
                f"{format_money(p25)} и {format_money(p75)}. "
                "Вакансии без вилки в расчёт не входят — их в базе заметная доля.",
            )
        )

    examples.append(
        (
            "Найди вакансии, где нужно чинить чужие сломанные автоматизации.",
            "Такое не находится по ключевым словам: в объявлениях пишут «поддержка "
            "интеграций» и «доработка существующих процессов». Это запрос к "
            "semantic_search — поиску по смыслу через pgvector.",
        )
    )
    return examples


def _percent(part: int, whole: int) -> str:
    if not whole:
        return "0%"
    return f"{part / whole * 100:.0f}%"
