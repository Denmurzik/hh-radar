"""Схема базы.

Три сущности и две связи. Вакансия принадлежит работодателю (many-to-one),
навыки связаны с вакансиями многие-ко-многим, чанки описания принадлежат
вакансии (one-to-many) и хранят вектор.

Отдельно стоит объяснить два решения, которые видно только в схеме:

1. ``Vacancy.search_vector`` — генерируемый столбец tsvector, а не вычисление
   на лету. Полнотекстовый индекс строится по нему, и Postgres не пересчитывает
   to_tsvector на каждый запрос. Заголовок вакансии весит больше описания
   (setweight A против B) — совпадение в названии релевантнее совпадения
   где-то в середине текста про ДМС.

2. ``Vacancy.salary_from_rub`` / ``salary_to_rub`` — денормализация. hh отдаёт
   зарплату в валюте вакансии, и сортировать смесь RUB/USD/KZT напрямую нельзя.
   Приведение к рублям делается на записи по статичному курсу
   (см. ``hh_radar.hh.parse.CURRENCY_RATES``) — это приближение, и оно честно
   помечено как приближение в README.
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

#: Размерность вектора paraphrase-multilingual-MiniLM-L12-v2. Вынесена в константу,
#: потому что
#: она зашита сразу в двух местах: в столбце Vector(...) и в миграции.
EMBEDDING_DIM = 384


class Base(DeclarativeBase):
    """Общий базовый класс для всех моделей."""


#: Связь многие-ко-многим между вакансиями и навыками.
#: Отдельной моделью не оформлена намеренно — у связи нет собственных атрибутов.
vacancy_skills = Table(
    "vacancy_skills",
    Base.metadata,
    Column(
        "vacancy_id",
        BigInteger,
        ForeignKey("vacancies.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "skill_id",
        Integer,
        ForeignKey("skills.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Index("ix_vacancy_skills_skill_id", "skill_id"),
)


class Employer(Base):
    """Работодатель. id — идентификатор hh, свой суррогатный ключ не нужен."""

    __tablename__ = "employers"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    alternate_url: Mapped[str | None] = mapped_column(String(512))
    trusted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    vacancies: Mapped[list[Vacancy]] = relationship(
        back_populates="employer", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_employers_name", "name"),)

    def __repr__(self) -> str:  # pragma: no cover - отладочное представление
        return f"<Employer id={self.id} name={self.name!r}>"


class Skill(Base):
    """Навык из key_skills вакансии.

    ``name`` — нормализованная форма (нижний регистр, схлопнутые пробелы,
    развёрнутые синонимы), по ней строится агрегация. ``display_name`` — то,
    как навык впервые написал работодатель, чтобы выдача выглядела человечно.
    """

    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False, unique=True)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)

    vacancies: Mapped[list[Vacancy]] = relationship(
        secondary=vacancy_skills, back_populates="skills"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Skill {self.name!r}>"


class Vacancy(Base):
    """Вакансия. id — идентификатор hh."""

    __tablename__ = "vacancies"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(String(512), nullable=False)

    employer_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("employers.id", ondelete="CASCADE")
    )
    employer: Mapped[Employer | None] = relationship(back_populates="vacancies")

    area_id: Mapped[int | None] = mapped_column(Integer)
    area_name: Mapped[str | None] = mapped_column(String(256))

    # Зарплата как её отдал hh.
    salary_from: Mapped[int | None] = mapped_column(Integer)
    salary_to: Mapped[int | None] = mapped_column(Integer)
    salary_currency: Mapped[str | None] = mapped_column(String(8))
    salary_gross: Mapped[bool | None] = mapped_column(Boolean)
    # Приведение к рублям для сортировки и сравнения (приближённое, см. docstring модуля).
    salary_from_rub: Mapped[int | None] = mapped_column(Integer)
    salary_to_rub: Mapped[int | None] = mapped_column(Integer)

    experience_id: Mapped[str | None] = mapped_column(String(64))
    experience_name: Mapped[str | None] = mapped_column(String(128))
    employment_id: Mapped[str | None] = mapped_column(String(64))
    schedule_id: Mapped[str | None] = mapped_column(String(64))
    is_remote: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    professional_roles: Mapped[list[dict] | None] = mapped_column(JSONB)

    description: Mapped[str | None] = mapped_column(Text)
    alternate_url: Mapped[str | None] = mapped_column(String(512))

    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Служебное: когда мы это видели и есть ли у нас полная карточка.
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    detail_fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    #: Генерируемый столбец: Postgres сам поддерживает его в актуальном состоянии.
    #: Название весит A, описание — B, поэтому ts_rank ставит совпадение
    #: в заголовке выше совпадения в теле объявления.
    search_vector: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('russian', coalesce(name, '')), 'A') || "
            "setweight(to_tsvector('russian', coalesce(description, '')), 'B')",
            persisted=True,
        ),
    )

    skills: Mapped[list[Skill]] = relationship(
        secondary=vacancy_skills, back_populates="vacancies"
    )
    chunks: Mapped[list[VacancyChunk]] = relationship(
        back_populates="vacancy", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        # Основной способ листать базу: свежие сверху.
        Index("ix_vacancies_published_at", published_at.desc()),
        # Полнотекстовый поиск.
        Index("ix_vacancies_search_vector", "search_vector", postgresql_using="gin"),
        # Частый фильтр «есть ли зарплата от N».
        Index("ix_vacancies_salary_from_rub", "salary_from_rub"),
        Index("ix_vacancies_employer_id", "employer_id"),
        # Витрина считает срезы по региону и опыту.
        Index("ix_vacancies_area_experience", "area_id", "experience_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Vacancy id={self.id} name={self.name!r}>"


class VacancyChunk(Base):
    """Кусок описания вакансии вместе с его эмбеддингом.

    Чанки лежат в той же базе, что и вакансии: pgvector — расширение Postgres,
    отдельная векторная СУБД для этого объёма не нужна.
    """

    __tablename__ = "vacancy_chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    vacancy_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("vacancies.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    char_len: Mapped[int] = mapped_column(Integer, nullable=False)

    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    embedded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    vacancy: Mapped[Vacancy] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("vacancy_id", "chunk_index", name="uq_chunk_position"),
        # HNSW по косинусному расстоянию: строится дольше ivfflat,
        # зато не требует предварительно набранных данных для обучения списков
        # и даёт лучший recall на этом объёме.
        Index(
            "ix_vacancy_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<VacancyChunk vacancy={self.vacancy_id} #{self.chunk_index}>"
