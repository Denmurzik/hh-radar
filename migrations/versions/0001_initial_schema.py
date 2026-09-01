"""Начальная схема: вакансии, работодатели, навыки, чанки с эмбеддингами.

Расширения создаются здесь, а не только в init-скрипте контейнера: в CI
поднимается голый service container, который init-скрипты из репозитория
не подхватывает, и без ``CREATE EXTENSION vector`` миграция там упадёт.

Revision ID: 0001
Revises:
Create Date: 2026-09-01
"""

from __future__ import annotations

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Совпадает с hh_radar.db.models.EMBEDDING_DIM.
#: Продублировано намеренно: миграция не должна меняться задним числом,
#: если в коде однажды поменяют модель.
EMBEDDING_DIM = 384


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.create_table(
        "employers",
        sa.Column("id", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column("alternate_url", sa.String(length=512), nullable=True),
        sa.Column("trusted", sa.Boolean(), nullable=False),
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_employers_name", "employers", ["name"], unique=False)

    op.create_table(
        "skills",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("display_name", sa.String(length=256), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    op.create_table(
        "vacancies",
        sa.Column("id", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column("employer_id", sa.BigInteger(), nullable=True),
        sa.Column("area_id", sa.Integer(), nullable=True),
        sa.Column("area_name", sa.String(length=256), nullable=True),
        sa.Column("salary_from", sa.Integer(), nullable=True),
        sa.Column("salary_to", sa.Integer(), nullable=True),
        sa.Column("salary_currency", sa.String(length=8), nullable=True),
        sa.Column("salary_gross", sa.Boolean(), nullable=True),
        sa.Column("salary_from_rub", sa.Integer(), nullable=True),
        sa.Column("salary_to_rub", sa.Integer(), nullable=True),
        sa.Column("experience_id", sa.String(length=64), nullable=True),
        sa.Column("experience_name", sa.String(length=128), nullable=True),
        sa.Column("employment_id", sa.String(length=64), nullable=True),
        sa.Column("schedule_id", sa.String(length=64), nullable=True),
        sa.Column("is_remote", sa.Boolean(), nullable=False),
        sa.Column("professional_roles", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("alternate_url", sa.String(length=512), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("archived", sa.Boolean(), nullable=False),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("detail_fetched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed(
                "setweight(to_tsvector('russian', coalesce(name, '')), 'A') || "
                "setweight(to_tsvector('russian', coalesce(description, '')), 'B')",
                persisted=True,
            ),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["employer_id"], ["employers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_vacancies_area_experience", "vacancies", ["area_id", "experience_id"], unique=False
    )
    op.create_index("ix_vacancies_employer_id", "vacancies", ["employer_id"], unique=False)
    op.create_index(
        "ix_vacancies_published_at",
        "vacancies",
        [sa.literal_column("published_at DESC")],
        unique=False,
    )
    op.create_index("ix_vacancies_salary_from_rub", "vacancies", ["salary_from_rub"], unique=False)
    op.create_index(
        "ix_vacancies_search_vector",
        "vacancies",
        ["search_vector"],
        unique=False,
        postgresql_using="gin",
    )

    op.create_table(
        "vacancy_chunks",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("vacancy_id", sa.BigInteger(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("char_len", sa.Integer(), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column(
            "embedded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["vacancy_id"], ["vacancies.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("vacancy_id", "chunk_index", name="uq_chunk_position"),
    )
    op.create_index(
        "ix_vacancy_chunks_embedding_hnsw",
        "vacancy_chunks",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "vacancy_skills",
        sa.Column("vacancy_id", sa.BigInteger(), nullable=False),
        sa.Column("skill_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["skill_id"], ["skills.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["vacancy_id"], ["vacancies.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("vacancy_id", "skill_id"),
    )
    op.create_index("ix_vacancy_skills_skill_id", "vacancy_skills", ["skill_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_vacancy_skills_skill_id", table_name="vacancy_skills")
    op.drop_table("vacancy_skills")
    op.drop_index("ix_vacancy_chunks_embedding_hnsw", table_name="vacancy_chunks")
    op.drop_table("vacancy_chunks")
    op.drop_index("ix_vacancies_search_vector", table_name="vacancies")
    op.drop_index("ix_vacancies_salary_from_rub", table_name="vacancies")
    op.drop_index("ix_vacancies_published_at", table_name="vacancies")
    op.drop_index("ix_vacancies_employer_id", table_name="vacancies")
    op.drop_index("ix_vacancies_area_experience", table_name="vacancies")
    op.drop_table("vacancies")
    op.drop_table("skills")
    op.drop_index("ix_employers_name", table_name="employers")
    op.drop_table("employers")
    # Расширения не удаляем: ими могут пользоваться другие схемы в той же базе.
