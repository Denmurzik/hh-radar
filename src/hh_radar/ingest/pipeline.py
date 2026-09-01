"""Конвейер: hh → база.

Устроен так, чтобы его можно было запускать хоть каждый час и не получать ни
дублей, ни потерь. Три решения, которые к этому ведут:

* **Идемпотентность через UPSERT.** Первичный ключ вакансии — идентификатор hh,
  а не автоинкремент. Повторный проход по тому же окну обновляет записи,
  а не плодит копии.
* **Двухфазная загрузка.** Поиск отдаёт до сотни вакансий за запрос, но без
  ``description`` и ``key_skills``; полная карточка стоит одного запроса на
  вакансию. Поэтому сначала пишется всё, что дал поиск, и только потом
  дозагружаются карточки — по вакансиям, у которых их ещё нет. Прерванный на
  середине запуск не теряет собранное: следующий подхватит с того же места.
* **Пакетные коммиты.** Коммит на каждую вакансию — это лишние тысячи
  round-trip'ов; один коммит на весь прогон — риск потерять час работы
  из-за одной битой записи.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import literal_column, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from hh_radar.db.models import Employer, Skill, Vacancy, vacancy_skills
from hh_radar.hh.client import HHClient, HHError, HHNotFoundError
from hh_radar.hh.parse import ParsedVacancy, normalize_skill_name, parse_vacancy

logger = logging.getLogger(__name__)

#: Сколько вакансий писать между коммитами.
BATCH_SIZE = 100


@dataclass
class IngestReport:
    """Что произошло за прогон. Печатается CLI и попадает в логи."""

    queries: list[str] = field(default_factory=list)
    seen: int = 0
    inserted: int = 0
    updated: int = 0
    details_fetched: int = 0
    details_failed: int = 0
    skills_linked: int = 0
    employers_touched: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    @property
    def elapsed_seconds(self) -> float:
        end = self.finished_at or datetime.now(UTC)
        return (end - self.started_at).total_seconds()

    def as_lines(self) -> list[str]:
        return [
            f"запросов: {len(self.queries)}",
            f"вакансий получено: {self.seen}",
            f"добавлено: {self.inserted}, обновлено: {self.updated}",
            f"карточек дозагружено: {self.details_fetched} (ошибок {self.details_failed})",
            f"работодателей: {self.employers_touched}, связей с навыками: {self.skills_linked}",
            f"время: {self.elapsed_seconds:.1f} с",
        ]


def ingest(
    session: Session,
    client: HHClient,
    *,
    queries: list[str],
    days: int = 30,
    area: int | str | None = None,
    fetch_details: bool = True,
    detail_limit: int | None = None,
    report: IngestReport | None = None,
) -> IngestReport:
    """Собрать вакансии по списку поисковых запросов и разложить по таблицам."""
    report = report or IngestReport()
    report.queries = list(queries)

    date_to = datetime.now(UTC)
    date_from = date_to - timedelta(days=days)

    for query in queries:
        logger.info("собираем «%s» за %d дней", query, days)
        batch: list[ParsedVacancy] = []
        for raw in client.iter_vacancies(
            text=query, date_from=date_from, date_to=date_to, area=area
        ):
            batch.append(parse_vacancy(raw))
            report.seen += 1
            if len(batch) >= BATCH_SIZE:
                _flush(session, batch, report)
                batch.clear()
        if batch:
            _flush(session, batch, report)

    if fetch_details:
        fetch_missing_details(session, client, limit=detail_limit, report=report)

    report.finished_at = datetime.now(UTC)
    return report


def fetch_missing_details(
    session: Session,
    client: HHClient,
    *,
    limit: int | None = None,
    report: IngestReport | None = None,
) -> IngestReport:
    """Дозагрузить полные карточки для вакансий, у которых их ещё нет.

    Отделено от :func:`ingest` намеренно: это самая долгая часть (один запрос
    на вакансию), и её полезно уметь запускать и останавливать отдельно.
    """
    report = report or IngestReport()

    stmt = (
        select(Vacancy.id)
        .where(Vacancy.detail_fetched_at.is_(None))
        .order_by(Vacancy.published_at.desc().nullslast())
    )
    if limit is not None:
        stmt = stmt.limit(limit)
    pending = list(session.scalars(stmt))

    logger.info("карточек к дозагрузке: %d", len(pending))
    for index, vacancy_id in enumerate(pending, start=1):
        try:
            raw = client.get_vacancy(vacancy_id)
        except HHNotFoundError:
            # Вакансию сняли. Помечаем архивной, чтобы не приходить за ней снова.
            session.query(Vacancy).filter(Vacancy.id == vacancy_id).update(
                {"archived": True, "detail_fetched_at": datetime.now(UTC)}
            )
            report.details_failed += 1
            continue
        except HHError as exc:
            logger.warning("не удалось получить карточку %s: %s", vacancy_id, exc)
            report.errors.append(f"vacancy {vacancy_id}: {exc}")
            report.details_failed += 1
            continue

        parsed = parse_vacancy(raw)
        _upsert_vacancy(session, parsed, report, mark_detailed=True)
        _link_skills(session, parsed, report)
        report.details_fetched += 1

        if index % BATCH_SIZE == 0:
            session.commit()

    session.commit()
    report.finished_at = datetime.now(UTC)
    return report


# --------------------------------------------------------------- internals --


def _flush(session: Session, batch: list[ParsedVacancy], report: IngestReport) -> None:
    for parsed in batch:
        _upsert_vacancy(session, parsed, report, mark_detailed=parsed.is_detailed)
        if parsed.skills:
            _link_skills(session, parsed, report)
    session.commit()


def _upsert_vacancy(
    session: Session, parsed: ParsedVacancy, report: IngestReport, *, mark_detailed: bool
) -> None:
    if parsed.employer is not None:
        _upsert_employer(session, parsed, report)

    now = datetime.now(UTC)
    values: dict[str, object] = {
        "id": parsed.id,
        "name": parsed.name,
        "employer_id": parsed.employer.id if parsed.employer else None,
        "area_id": parsed.area_id,
        "area_name": parsed.area_name,
        "salary_from": parsed.salary_from,
        "salary_to": parsed.salary_to,
        "salary_currency": parsed.salary_currency,
        "salary_gross": parsed.salary_gross,
        "salary_from_rub": parsed.salary_from_rub,
        "salary_to_rub": parsed.salary_to_rub,
        "experience_id": parsed.experience_id,
        "experience_name": parsed.experience_name,
        "employment_id": parsed.employment_id,
        "schedule_id": parsed.schedule_id,
        "is_remote": parsed.is_remote,
        "professional_roles": parsed.professional_roles or None,
        "alternate_url": parsed.alternate_url,
        "published_at": parsed.published_at,
        "created_at": parsed.created_at,
        "archived": parsed.archived,
        "fetched_at": now,
    }
    if parsed.description is not None:
        values["description"] = parsed.description
    if mark_detailed:
        values["detail_fetched_at"] = now

    stmt = insert(Vacancy).values(**values)
    # Поисковая выдача не содержит description — и не должна затирать уже
    # загруженное полное описание пустым значением. Поэтому обновляются
    # только те поля, которые в этом ответе реально пришли.
    update_columns = {key: getattr(stmt.excluded, key) for key in values if key != "id"}
    stmt = stmt.on_conflict_do_update(index_elements=[Vacancy.id], set_=update_columns)

    # xmax системного столбца равен нулю у только что вставленной строки и
    # содержит идентификатор транзакции у обновлённой. Это штатный способ
    # отличить INSERT от UPDATE внутри ON CONFLICT, не делая лишнего SELECT.
    row = session.execute(
        stmt.returning(literal_column("(xmax = 0)").label("was_inserted"))
    ).first()
    if row is None:
        return
    if row.was_inserted:
        report.inserted += 1
    else:
        report.updated += 1


def _upsert_employer(session: Session, parsed: ParsedVacancy, report: IngestReport) -> None:
    employer = parsed.employer
    if employer is None:
        return
    stmt = insert(Employer).values(
        id=employer.id,
        name=employer.name,
        alternate_url=employer.alternate_url,
        trusted=employer.trusted,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[Employer.id],
        set_={
            "name": stmt.excluded.name,
            "alternate_url": stmt.excluded.alternate_url,
            "trusted": stmt.excluded.trusted,
        },
    )
    session.execute(stmt)
    report.employers_touched += 1


def _link_skills(session: Session, parsed: ParsedVacancy, report: IngestReport) -> None:
    """Записать навыки вакансии и связи с ней.

    Связи перезаписываются целиком: работодатель мог убрать навык из вакансии,
    и оставлять устаревшую связь — значит врать в статистике.
    """
    if not parsed.skills:
        return

    skill_ids: list[int] = []
    for raw_name in parsed.skills:
        normalized = normalize_skill_name(raw_name)
        if not normalized:
            continue
        stmt = insert(Skill).values(name=normalized, display_name=raw_name)
        # DO UPDATE вместо DO NOTHING: только он гарантированно возвращает id
        # и для вставки, и для конфликта.
        stmt = stmt.on_conflict_do_update(
            index_elements=[Skill.name], set_={"name": stmt.excluded.name}
        )
        skill_id = session.scalar(stmt.returning(Skill.id))
        if skill_id is not None:
            skill_ids.append(skill_id)

    session.execute(vacancy_skills.delete().where(vacancy_skills.c.vacancy_id == parsed.id))
    if skill_ids:
        session.execute(
            insert(vacancy_skills)
            .values([{"vacancy_id": parsed.id, "skill_id": sid} for sid in skill_ids])
            .on_conflict_do_nothing()
        )
        report.skills_linked += len(skill_ids)
