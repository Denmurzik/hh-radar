"""Тесты семантического поиска без базы данных и без эмбеддинг-модели.

Проверяем чистые функции, из которых собран search.py: схлопывание чанков по
вакансии, RRF и сериализацию SemanticHit. Ничего из этого не требует
подключения к Postgres. Маркеры integration/embeddings не нужны.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from hh_radar.rag.search import (
    SemanticHit,
    _collapse_best_chunk_per_vacancy,
    reciprocal_rank_fusion,
)


def test_semantic_hit_to_dict_is_json_compatible_with_iso_date() -> None:
    hit = SemanticHit(
        vacancy_id=123,
        name="Python-разработчик",
        employer_name="ООО Ромашка",
        similarity=0.87,
        chunk_index=1,
        snippet="Ищем python-разработчика в команду автоматизации.",
        alternate_url="https://hh.ru/vacancy/123",
        published_at=datetime(2026, 8, 15, 12, 30, tzinfo=UTC),
    )
    data = hit.to_dict()

    assert data["vacancy_id"] == 123
    assert data["name"] == "Python-разработчик"
    assert data["employer_name"] == "ООО Ромашка"
    assert data["similarity"] == 0.87
    assert data["chunk_index"] == 1
    assert data["alternate_url"] == "https://hh.ru/vacancy/123"
    # datetime сериализован в ISO-строку, а не остался объектом datetime
    assert data["published_at"] == "2026-08-15T12:30:00+00:00"
    assert isinstance(data["published_at"], str)


def test_semantic_hit_to_dict_handles_missing_published_at() -> None:
    hit = SemanticHit(
        vacancy_id=1,
        name="Вакансия",
        employer_name=None,
        similarity=0.5,
        chunk_index=0,
        snippet="",
        alternate_url=None,
        published_at=None,
    )
    assert hit.to_dict()["published_at"] is None


@dataclass(frozen=True, slots=True)
class _FakeChunkRow:
    """Заглушка строки результата вместо реального SQLAlchemy Row —
    достаточно атрибутов vacancy_id/distance, которые использует
    _collapse_best_chunk_per_vacancy."""

    vacancy_id: int
    distance: float


def test_collapse_best_chunk_keeps_only_best_chunk_per_vacancy() -> None:
    rows = [
        _FakeChunkRow(vacancy_id=1, distance=0.4),  # similarity 0.6
        _FakeChunkRow(vacancy_id=1, distance=0.1),  # similarity 0.9 — лучший для вакансии 1
        _FakeChunkRow(vacancy_id=1, distance=0.3),  # similarity 0.7
        _FakeChunkRow(vacancy_id=2, distance=0.2),  # similarity 0.8 — единственный для вакансии 2
    ]
    best = _collapse_best_chunk_per_vacancy(rows, min_similarity=0.0)

    assert set(best) == {1, 2}
    similarity_v1, row_v1 = best[1]
    assert row_v1.distance == 0.1
    assert similarity_v1 == 0.9
    similarity_v2, _row_v2 = best[2]
    assert similarity_v2 == 0.8


def test_collapse_best_chunk_respects_min_similarity() -> None:
    rows = [
        _FakeChunkRow(vacancy_id=1, distance=0.1),  # similarity 0.9
        _FakeChunkRow(vacancy_id=2, distance=0.8),  # similarity 0.2 — должна отсечься
    ]
    best = _collapse_best_chunk_per_vacancy(rows, min_similarity=0.5)

    assert set(best) == {1}


def test_reciprocal_rank_fusion_combines_two_rankings() -> None:
    semantic_rank = {10: 1, 20: 2, 30: 3}
    fulltext_rank = {20: 1, 10: 2}

    scores = reciprocal_rank_fusion([semantic_rank, fulltext_rank], [0.5, 0.5], k=60)

    # вакансия 20 первая в fulltext и вторая в semantic — она должна обойти
    # вакансию 10, которая первая в semantic, но вторая в fulltext, т.к. обе
    # найдены обоими методами с почти симметричными рангами (1+2 vs 2+1),
    # а формула симметрична относительно перестановки методов при равном весе.
    assert scores[10] == 0.5 / (60 + 1) + 0.5 / (60 + 2)
    assert scores[20] == 0.5 / (60 + 2) + 0.5 / (60 + 1)
    assert scores[10] == scores[20]

    # вакансия 30 найдена только семантическим методом — вклад только от него
    assert scores[30] == 0.5 / (60 + 3)
    assert 30 in scores and len(scores) == 3


def test_reciprocal_rank_fusion_respects_weights() -> None:
    semantic_rank = {1: 1}
    fulltext_rank = {2: 1}

    scores = reciprocal_rank_fusion([semantic_rank, fulltext_rank], [0.9, 0.1], k=60)

    assert scores[1] > scores[2]
    assert scores[1] == 0.9 / 61
    assert scores[2] == 0.1 / 61


def test_reciprocal_rank_fusion_empty_rankings_give_empty_scores() -> None:
    assert reciprocal_rank_fusion([{}, {}], [0.5, 0.5]) == {}
