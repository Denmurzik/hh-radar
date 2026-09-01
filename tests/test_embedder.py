"""Тесты эмбеддера: HashEmbedder, фабрика get_embedder, ленивая загрузка fastembed.

Всё на backend="hash" — реальная модель fastembed нигде не скачивается и не
запускается. Маркеры integration/embeddings не нужны.
"""

from __future__ import annotations

from hh_radar.config import Settings
from hh_radar.rag.embedder import (
    FastEmbedEmbedder,
    HashEmbedder,
    cosine_similarity,
    get_embedder,
    reset_embedder_cache,
)


def test_hash_embedder_is_deterministic() -> None:
    embedder = HashEmbedder(dim=384)
    v1 = embedder.embed_query("python разработчик автоматизация процессов")
    v2 = embedder.embed_query("python разработчик автоматизация процессов")
    assert v1 == v2


def test_hash_embedder_has_correct_dimension() -> None:
    embedder = HashEmbedder(dim=384)
    vector = embedder.embed_query("тестовый запрос")
    assert len(vector) == 384

    docs = embedder.embed_documents(["первый текст", "второй текст"])
    assert len(docs) == 2
    assert all(len(v) == 384 for v in docs)


def test_hash_embedder_vectors_are_l2_normalized() -> None:
    embedder = HashEmbedder(dim=384)
    vector = embedder.embed_query("любой непустой текст для проверки нормы")
    norm = sum(x * x for x in vector) ** 0.5
    assert abs(norm - 1.0) < 1e-9


def test_hash_embedder_similar_texts_are_closer_than_unrelated() -> None:
    embedder = HashEmbedder(dim=384)
    base = embedder.embed_query("python разработчик автоматизация интеграций")
    related = embedder.embed_query("python разработчик для автоматизации бизнес-процессов")
    unrelated = embedder.embed_query("повар в ресторане на кухне ищет работу")

    sim_related = cosine_similarity(base, related)
    sim_unrelated = cosine_similarity(base, unrelated)
    assert sim_related > sim_unrelated


def test_get_embedder_returns_hash_backend() -> None:
    reset_embedder_cache()
    settings = Settings(EMBEDDING_BACKEND="hash")
    embedder = get_embedder(settings)
    assert isinstance(embedder, HashEmbedder)
    reset_embedder_cache()


def test_get_embedder_caches_instance() -> None:
    reset_embedder_cache()
    settings = Settings(EMBEDDING_BACKEND="hash")
    first = get_embedder(settings)
    second = get_embedder(settings)
    assert first is second
    reset_embedder_cache()


def test_fastembed_embedder_does_not_load_model_on_construction() -> None:
    # Модель НЕ скачиваем и НЕ создаём: проверяем только то, что конструктор
    # и импорт модуля не трогают fastembed.TextEmbedding.
    embedder = FastEmbedEmbedder(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        dim=384,
    )
    assert embedder._model is None
    assert embedder.dim == 384
    assert embedder.model_name == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
