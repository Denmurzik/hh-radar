"""Тесты настоящей модели эмбеддингов.

Отделены от остальных маркером ``embeddings``: первый запуск выкачивает
~220 МБ ONNX, поэтому в CI и в обычном прогоне их нет.

Проверяется ровно то, ради чего модель в проекте и стоит: что она понимает
русский перифраз. Заглушка ``HashEmbedder``, на которой работают остальные
тесты, этого не умеет и уметь не должна — она мешок слов. Если бы такой
проверки не было, подмена настоящей модели заглушкой прошла бы незамеченной:
весь остальной код от этого не падает, просто поиск по смыслу перестаёт
искать по смыслу.

    uv run pytest -m embeddings
"""

from __future__ import annotations

import pytest

from hh_radar.config import Settings
from hh_radar.rag.embedder import cosine_similarity, get_embedder, reset_embedder_cache

pytestmark = pytest.mark.embeddings


@pytest.fixture
def embedder() -> object:
    reset_embedder_cache()
    settings = Settings(embedding_backend="fastembed")
    try:
        return get_embedder(settings)
    except Exception as exc:  # pragma: no cover — нет сети и нет кэша модели
        pytest.skip(f"модель недоступна: {exc}")


def test_dimension_matches_the_database_column(embedder) -> None:  # type: ignore[no-untyped-def]
    """384 зашито в схеме: при несовпадении запись вектора упадёт в базе."""
    assert len(embedder.embed_query("инженер по автоматизации")) == 384


def test_paraphrase_is_closer_than_a_different_profession(embedder) -> None:  # type: ignore[no-untyped-def]
    """Главное свойство модели: близость по смыслу, а не по словам.

    У перифраза с эталоном нет ни одного общего значимого слова — на мешке
    слов он проиграл бы вакансии повара, где общее слово «требуется» есть.
    """
    reference = embedder.embed_documents(
        [
            "Инженер по автоматизации: сопровождение и доработка интеграций",
            "Повар горячего цеха в ресторан, требуется опыт работы на банкетах",
        ]
    )
    query = embedder.embed_query("требуется чинить чужие сломанные сценарии обмена данными")

    to_automation = cosine_similarity(query, reference[0])
    to_cook = cosine_similarity(query, reference[1])
    assert to_automation > to_cook
