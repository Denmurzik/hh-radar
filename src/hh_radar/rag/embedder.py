"""Абстракция эмбеддера и две реализации: настоящая модель и hash-заглушка.

Протокол ``Embedder`` даёт индексатору и поиску единый интерфейс независимо
от того, какой бэкенд выбран в конфиге через ``settings.embedding_backend``.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from hh_radar.config import Settings, get_settings


@runtime_checkable
class Embedder(Protocol):
    """Общий интерфейс эмбеддера: реальная модель и hash-заглушка взаимозаменяемы."""

    dim: int
    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Эмбеддинги для индексации, батчем."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Эмбеддинг одного поискового запроса."""
        ...


class FastEmbedEmbedder:
    """Обёртка над ``fastembed.TextEmbedding``.

    Модель (ONNX-веса, для paraphrase-multilingual-MiniLM-L12-v2 — ~220 МБ)
    загружается ЛЕНИВО: импорт ``fastembed`` и создание сессии инференса
    откладываются до первого реального вызова ``embed_documents``/``embed_query``.
    Это важно, потому что ``from hh_radar.rag import embedder`` (например, из
    CLI-команды полнотекстового поиска или из тестов на backend="hash") не
    должен тянуть скачивание модели там, где эмбеддинги вообще не нужны.
    """

    def __init__(self, model_name: str, dim: int, cache_dir: str | None = None) -> None:
        self.model_name = model_name
        self.dim = dim
        self._cache_dir = cache_dir
        # None, пока модель не понадобилась ни разу — см. docstring класса.
        self._model: object | None = None

    def _get_model(self) -> object:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=self._cache_dir,
                lazy_load=True,
            )
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        vectors = model.embed(list(texts))  # type: ignore[attr-defined]
        # fastembed отдаёт numpy-массивы; в базу (pgvector.sqlalchemy.Vector)
        # кладём обычные списки float, чтобы не тащить numpy в слой хранения.
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class HashEmbedder:
    """Детерминированная заглушка эмбеддера без ML-модели.

    Это осознанное инженерное решение, а не халтура: ``HashEmbedder`` нужен
    не ради качества векторов, а ради того, чтобы тесты, CI и
    ``docker compose up`` работали без скачивания ~220 МБ ONNX-модели и без
    инференса на CPU при каждом запуске. В проде (``embedding_backend`` по
    умолчанию — ``"fastembed"``) он не используется — переключение бэкендов
    делает ``get_embedder``.

    Алгоритм — мешок слов, хешированный в фиксированную размерность:
    текст токенизируется на слова, каждое слово через blake2b детерминированно
    попадает в одну из ``dim`` позиций вектора со знаком, тоже определяемым
    хешем, вклады слов суммируются, итоговый вектор нормируется по L2. Один и
    тот же текст всегда даёт один и тот же вектор (никакой случайности), а
    тексты с общими словами дают более близкий по косинусу результат, чем
    тексты без общих слов — этого достаточно, чтобы протестировать код поиска
    и ранжирования (top-k, схлопывание по вакансии, RRF) без настоящей
    семантики.
    """

    def __init__(self, dim: int = 384, model_name: str = "hash-bow-v1") -> None:
        self.dim = dim
        self.model_name = model_name

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]
        return vector

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


_embedder_cache: dict[str, Embedder] = {}


def _cache_key(settings: Settings) -> str:
    return f"{settings.embedding_backend}:{settings.embedding_model}:{settings.embedding_dim}"


def get_embedder(settings: Settings | None = None) -> Embedder:
    """Фабрика эмбеддера по ``settings.embedding_backend``.

    Кэширует экземпляр по (backend, model, dim): для ``FastEmbedEmbedder`` это
    значит, что модель, даже будучи лениво загруженной, грузится в память один
    раз за процесс, а не при каждом вызове ``get_embedder``.
    """
    settings = settings or get_settings()
    key = _cache_key(settings)
    embedder = _embedder_cache.get(key)
    if embedder is None:
        if settings.embedding_backend == "hash":
            embedder = HashEmbedder(dim=settings.embedding_dim)
        else:
            embedder = FastEmbedEmbedder(
                model_name=settings.embedding_model,
                dim=settings.embedding_dim,
                cache_dir=str(settings.embedding_cache_dir),
            )
        _embedder_cache[key] = embedder
    return embedder


def reset_embedder_cache() -> None:
    """Сбросить кэш экземпляров эмбеддера. Нужно тестам, меняющим backend/модель."""
    _embedder_cache.clear()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Косинусное сходство двух векторов на чистом Python.

    В боевом поиске сравнение векторов делает pgvector прямо в SQL
    (``VacancyChunk.embedding.cosine_distance``), но для юнит-тестов и для
    сравнения hash-векторов удобнее не тянуть numpy или базу.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
