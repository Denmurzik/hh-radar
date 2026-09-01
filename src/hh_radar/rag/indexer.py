"""Индексация вакансий: описание -> чанки -> эмбеддинги -> vacancy_chunks."""

from __future__ import annotations

import time
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from hh_radar.config import get_settings
from hh_radar.db.models import Vacancy, VacancyChunk
from hh_radar.rag.chunking import chunk_vacancy
from hh_radar.rag.embedder import Embedder, get_embedder


@dataclass(frozen=True, slots=True)
class IndexReport:
    """Итог одного запуска индексации."""

    vacancies_processed: int
    chunks_written: int
    chunks_skipped: int
    model_name: str
    elapsed_seconds: float


def index_vacancies(
    session: Session,
    *,
    embedder: Embedder | None = None,
    batch_size: int = 64,
    only_missing: bool = True,
    limit: int | None = None,
) -> IndexReport:
    """Режет описания вакансий на чанки и считает эмбеддинги.

    Идемпотентность при ``only_missing=True``: вакансия целиком пропускается
    (``chunks_skipped`` += 1), если у неё уже есть хотя бы один чанк с
    текущим ``embedder.model_name`` — значит, эта вакансия этой моделью уже
    проиндексирована, повторный проход не нужен. Если у вакансии есть чанки,
    но от ДРУГОЙ модели (модель в конфиге поменялась), они удаляются и
    считаются заново — в базе никогда не остаётся смесь векторов из разных
    моделей для одной вакансии, потому что расстояния между эмбеддингами
    разных моделей несравнимы. При ``only_missing=False`` (принудительный
    reindex) чанки вакансии тоже сначала удаляются, потом пересчитываются —
    иначе новая вставка упёрлась бы в уникальный индекс (vacancy_id, chunk_index).

    Список вакансий для обработки материализуется одним запросом (``id``,
    ``name``, ``description`` — компактные поля, без векторов), а не через
    потоковый курсор (``yield_per``): в SQLAlchemy/psycopg серверный курсор
    привязан к транзакции, и периодический ``session.commit()`` внутри цикла
    (см. ниже) закрыл бы его раньше времени. Тяжёлая часть — тексты чанков и
    их векторы — в памяти целиком не держится: они считаются и пишутся
    пачками по ``batch_size``.
    """
    started = time.monotonic()
    settings = get_settings()
    embedder = embedder or get_embedder(settings)

    query = select(Vacancy).where(Vacancy.description.is_not(None), Vacancy.description != "")
    if limit is not None:
        query = query.limit(limit)
    vacancies = session.execute(query).scalars().all()

    # Модели, которыми уже проиндексирована каждая вакансия — одним запросом
    # на все вакансии сразу, а не по одному SELECT на вакансию в цикле.
    vacancy_ids = [v.id for v in vacancies]
    models_by_vacancy: dict[int, set[str]] = {}
    if vacancy_ids:
        pairs = session.execute(
            select(VacancyChunk.vacancy_id, VacancyChunk.model_name)
            .where(VacancyChunk.vacancy_id.in_(vacancy_ids))
            .distinct()
        ).all()
        for vacancy_id, model_name in pairs:
            models_by_vacancy.setdefault(vacancy_id, set()).add(model_name)

    vacancies_processed = 0
    chunks_written = 0
    chunks_skipped = 0

    pending_chunks: list[VacancyChunk] = []
    pending_texts: list[str] = []

    def flush_pending() -> None:
        nonlocal chunks_written
        if not pending_chunks:
            return
        vectors = embedder.embed_documents(pending_texts)
        for chunk, vector in zip(pending_chunks, vectors, strict=True):
            chunk.embedding = vector
        session.add_all(pending_chunks)
        # Коммитим пачками, а не по одной записи и не одной гигантской
        # транзакцией в конце: это и ограничивает память под незакоммиченные
        # объекты, и не даёт индексации многотысячной базы жить в одной
        # транзакции, которую откатит любая сетевая заминка.
        session.commit()
        chunks_written += len(pending_chunks)
        pending_chunks.clear()
        pending_texts.clear()

    for vacancy in vacancies:
        existing_models = models_by_vacancy.get(vacancy.id, set())
        if only_missing and embedder.model_name in existing_models:
            chunks_skipped += 1
            continue

        if existing_models:
            session.execute(delete(VacancyChunk).where(VacancyChunk.vacancy_id == vacancy.id))

        texts = chunk_vacancy(
            vacancy.name,
            vacancy.description or "",
            chunk_chars=settings.chunk_chars,
            overlap_chars=settings.chunk_overlap_chars,
        )
        vacancies_processed += 1

        for index, text in enumerate(texts):
            pending_chunks.append(
                VacancyChunk(
                    vacancy_id=vacancy.id,
                    chunk_index=index,
                    content=text,
                    char_len=len(text),
                    model_name=embedder.model_name,
                )
            )
            pending_texts.append(text)
            if len(pending_chunks) >= batch_size:
                flush_pending()

    flush_pending()

    return IndexReport(
        vacancies_processed=vacancies_processed,
        chunks_written=chunks_written,
        chunks_skipped=chunks_skipped,
        model_name=embedder.model_name,
        elapsed_seconds=time.monotonic() - started,
    )
