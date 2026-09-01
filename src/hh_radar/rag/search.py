"""Семантический и гибридный поиск по вакансиям."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from hh_radar.config import get_settings
from hh_radar.db.models import Employer, Vacancy, VacancyChunk
from hh_radar.db.queries import VacancySummary, search_vacancies
from hh_radar.rag.embedder import get_embedder


@dataclass(frozen=True, slots=True)
class SemanticHit:
    """Один результат семантического поиска — вакансия и её лучший чанк."""

    vacancy_id: int
    name: str
    employer_name: str | None
    similarity: float
    chunk_index: int
    snippet: str
    alternate_url: str | None
    published_at: datetime | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-совместимое представление (datetime -> ISO-строка):
        MCP-сервер отдаёт ответ инструмента как JSON."""
        return {
            "vacancy_id": self.vacancy_id,
            "name": self.name,
            "employer_name": self.employer_name,
            "similarity": self.similarity,
            "chunk_index": self.chunk_index,
            "snippet": self.snippet,
            "alternate_url": self.alternate_url,
            "published_at": self.published_at.isoformat() if self.published_at else None,
        }


def _make_snippet(content: str, max_chars: int = 400) -> str:
    """Обрезает содержимое чанка для показа в выдаче — сам эмбеддинг считается
    по полному чанку, а пользователю показывать несколько абзацев незачем."""
    content = content.strip()
    if len(content) <= max_chars:
        return content
    return content[:max_chars].rstrip() + "…"


def _collapse_best_chunk_per_vacancy(
    rows: Sequence[Any], *, min_similarity: float
) -> dict[int, tuple[float, Any]]:
    """Из чанков (обычно несколько на вакансию — их и было запрошено с запасом)
    оставляет по каждой вакансии только самый похожий чанк.

    Вынесена отдельной функцией от ``semantic_search``, чтобы это поведение
    можно было проверить в юнит-тесте на простых объектах-заглушках с полями
    ``vacancy_id``/``distance``, без базы данных.
    """
    best: dict[int, tuple[float, Any]] = {}
    for row in rows:
        similarity = 1.0 - float(row.distance)
        if similarity < min_similarity:
            continue
        current = best.get(row.vacancy_id)
        if current is None or similarity > current[0]:
            best[row.vacancy_id] = (similarity, row)
    return best


def semantic_search(
    session: Session, query: str, *, limit: int = 10, min_similarity: float = 0.0
) -> list[SemanticHit]:
    """Ищет вакансии по смыслу запроса через косинусное расстояние pgvector.

    Берём с запасом ``limit * 4`` ближайших ЧАНКОВ (не вакансий), а затем
    схлопываем результат по ``vacancy_id``, оставляя для каждой вакансии
    только лучший чанк (см. ``_collapse_best_chunk_per_vacancy``). Без этого
    приёма топ выдачи мог бы целиком состоять из чанков одной длинной
    вакансии с многословным описанием — у неё просто больше чанков, а значит
    больше шансов, что какой-то из них случайно окажется близко к запросу.
    Запас "*4" компенсирует схлопывание: после дедупликации по вакансии
    результатов должно остаться не меньше ``limit``, если в базе действительно
    есть столько релевантных вакансий.
    """
    settings = get_settings()
    embedder = get_embedder(settings)
    query_vector = embedder.embed_query(query)

    fetch_limit = max(limit * 4, limit)
    distance = VacancyChunk.embedding.cosine_distance(query_vector)
    stmt = (
        select(
            VacancyChunk.vacancy_id.label("vacancy_id"),
            VacancyChunk.chunk_index.label("chunk_index"),
            VacancyChunk.content.label("content"),
            distance.label("distance"),
            Vacancy.name.label("name"),
            Vacancy.alternate_url.label("alternate_url"),
            Vacancy.published_at.label("published_at"),
            Employer.name.label("employer_name"),
        )
        .select_from(VacancyChunk)
        .join(Vacancy, Vacancy.id == VacancyChunk.vacancy_id)
        .outerjoin(Employer, Employer.id == Vacancy.employer_id)
        .where(VacancyChunk.embedding.is_not(None))
        .order_by(distance)
        .limit(fetch_limit)
    )
    rows = session.execute(stmt).all()

    best_by_vacancy = _collapse_best_chunk_per_vacancy(rows, min_similarity=min_similarity)
    ranked = sorted(best_by_vacancy.values(), key=lambda item: item[0], reverse=True)[:limit]

    return [
        SemanticHit(
            vacancy_id=row.vacancy_id,
            name=row.name,
            employer_name=row.employer_name,
            similarity=similarity,
            chunk_index=row.chunk_index,
            snippet=_make_snippet(row.content),
            alternate_url=row.alternate_url,
            published_at=row.published_at,
        )
        for similarity, row in ranked
    ]


def reciprocal_rank_fusion(
    rankings: Sequence[Mapping[int, int]], weights: Sequence[float], *, k: int = 60
) -> dict[int, float]:
    """Чистая функция RRF: по нескольким ранжированиям и их весам считает
    сплавленный скор каждого id.

    ``rankings[i]`` — словарь ``{id: rank}`` (rank 1-based, 1 — лучший
    результат метода i); id, которого метод не нашёл, в его словаре просто
    нет и не даёт вклада в сумму.

        score(id) = sum_i weight_i / (k + rank_i(id))

    ``k=60`` — константа из оригинальной статьи про RRF (Cormack et al.,
    2009): сглаживает вклад позиций в начале рейтинга, не давая рангу 1 в
    одном методе задавить всё остальное. Вынесена отдельной чистой функцией
    (без обращений к базе), чтобы формулу можно было проверить в юнит-тесте
    напрямую, без эмбеддера и без запущенного Postgres.
    """
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for item_id, rank in ranking.items():
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank)
    return scores


def extract_vacancy_id(item: VacancySummary | SemanticHit) -> int:
    """Идентификатор вакансии из результата любого из двух поисков.

    Полнотекстовый поиск отдаёт ``VacancySummary`` с полем ``id``,
    семантический — ``SemanticHit`` с полем ``vacancy_id``. Сплавлять их
    ранги можно только после приведения к общему ключу.
    """
    return item.vacancy_id if isinstance(item, SemanticHit) else item.id


def _fetch_fallback_hit(session: Session, vacancy_id: int) -> SemanticHit:
    """Строит ``SemanticHit`` для вакансии, найденной только полнотекстовым
    поиском — её не было среди ``fetch_limit`` ближайших семантических чанков,
    поэтому у неё нет ни similarity, ни готового сниппета."""
    row = session.execute(
        select(
            Vacancy.name,
            Vacancy.alternate_url,
            Vacancy.published_at,
            Employer.name.label("employer_name"),
        )
        .select_from(Vacancy)
        .outerjoin(Employer, Employer.id == Vacancy.employer_id)
        .where(Vacancy.id == vacancy_id)
    ).first()
    if row is None:
        # вакансию успели удалить между полнотекстовым запросом и этим моментом
        return SemanticHit(
            vacancy_id=vacancy_id,
            name="",
            employer_name=None,
            similarity=0.0,
            chunk_index=-1,
            snippet="",
            alternate_url=None,
            published_at=None,
        )
    return SemanticHit(
        vacancy_id=vacancy_id,
        name=row.name,
        employer_name=row.employer_name,
        similarity=0.0,
        chunk_index=-1,
        snippet="",
        alternate_url=row.alternate_url,
        published_at=row.published_at,
    )


def hybrid_search(
    session: Session, query: str, *, limit: int = 10, semantic_weight: float = 0.5
) -> list[SemanticHit]:
    """Гибридный поиск: сплавляет ранжирование полнотекстового и
    семантического поиска через Reciprocal Rank Fusion (см.
    ``reciprocal_rank_fusion``).

    RRF выбран вместо взвешенной суммы "сырых" скоров, потому что скоры из
    разных пространств несравнимы: ``ts_rank`` полнотекстового поиска ничем
    не ограничен сверху и зависит от длины документа и частоты термина, а
    косинусное сходство лежит в [-1, 1] и означает совсем другую вещь.
    Складывать их напрямую — всё равно что складывать рубли с процентами.
    Ранг же универсален: и там, и там ранг 1 значит "лучший результат этого
    конкретного метода", и это единственное, что нужно RRF.

    """
    fetch_limit = max(limit * 4, limit)

    semantic_hits = semantic_search(session, query, limit=fetch_limit)
    semantic_rank = {hit.vacancy_id: i + 1 for i, hit in enumerate(semantic_hits)}
    hits_by_id = {hit.vacancy_id: hit for hit in semantic_hits}

    fulltext_results = search_vacancies(session, query, limit=fetch_limit)
    fulltext_rank = {extract_vacancy_id(item): i + 1 for i, item in enumerate(fulltext_results)}

    fused = reciprocal_rank_fusion(
        [semantic_rank, fulltext_rank], [semantic_weight, 1 - semantic_weight]
    )
    top_ids = sorted(fused, key=lambda vacancy_id: fused[vacancy_id], reverse=True)[:limit]

    return [
        hits_by_id[vacancy_id]
        if vacancy_id in hits_by_id
        else _fetch_fallback_hit(session, vacancy_id)
        for vacancy_id in top_ids
    ]
