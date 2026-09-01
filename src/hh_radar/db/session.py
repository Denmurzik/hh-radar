"""Подключение к базе.

Один Engine на процесс, сессии — короткоживущие. MCP-сервер обслуживает
запросы агента последовательно, поэтому пул маленький: держать двадцать
соединений ради одного stdio-клиента незачем.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from hh_radar.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Engine-синглтон. ``pool_pre_ping`` — чтобы переживать перезапуск контейнера."""
    settings = get_settings()
    return create_engine(
        settings.database_url,
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
        future=True,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Сессия с транзакцией: коммит на успешном выходе, откат на исключении."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ping() -> bool:
    """Проверка, что база отвечает. Используется в CLI и в healthcheck-команде."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def reset_caches() -> None:
    """Сбросить закэшированные engine и фабрику. Нужно тестам, меняющим DSN."""
    get_engine.cache_clear()
    get_session_factory.cache_clear()
