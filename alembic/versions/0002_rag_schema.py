"""rag schema: pgvector extension, document, chunk (+HNSW), service tables

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql as pg

from trt.config import get_settings

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

DIM = get_settings().embedding_dim


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE SCHEMA IF NOT EXISTS rag")

    op.create_table(
        "document",
        sa.Column("document_id", sa.BigInteger, sa.Identity(), primary_key=True),
        sa.Column("doc_key", sa.String(512), nullable=False, unique=True),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column(
            "text_id", sa.BigInteger, sa.ForeignKey("i18n_text.text_id", ondelete="SET NULL")
        ),
        sa.Column("text_code", sa.String(256)),
        sa.Column("owner_kind", sa.String(20)),
        sa.Column("locale_code", sa.String(16), sa.ForeignKey("i18n_locale.locale_code")),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("pages", pg.JSONB),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("raw_path", sa.Text),
        sa.Column("source_updated_at", sa.DateTime(timezone=True)),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("chunked_sha256", sa.String(64)),
        sa.CheckConstraint("kind in ('ui_string', 'reference_doc')", name="ck_document_kind"),
        schema="rag",
    )
    op.create_index("ix_document_text_locale", "document", ["text_id", "locale_code"], schema="rag")
    op.create_index("ix_document_text_code", "document", ["text_code"], schema="rag")
    op.create_index("ix_document_sha", "document", ["content_sha256"], schema="rag")
    op.create_index(
        "ix_document_source_upd",
        "document",
        ["source", sa.text("source_updated_at DESC")],
        schema="rag",
    )
    op.execute(
        "CREATE INDEX ix_document_fts ON rag.document USING gin (to_tsvector('simple', content))"
    )

    op.create_table(
        "chunk",
        sa.Column("chunk_id", sa.BigInteger, sa.Identity(), primary_key=True),
        sa.Column(
            "document_id",
            sa.BigInteger,
            sa.ForeignKey("rag.document.document_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("embed_text", sa.Text, nullable=False),
        sa.Column("embedding", Vector(DIM), nullable=False),
        sa.Column("embedding_model", sa.String(128), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("locale_code", sa.String(16)),
        sa.Column("text_id", sa.BigInteger),
        sa.Column("owner_kind", sa.String(20)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunk_document_index"),
        schema="rag",
    )
    op.create_index("ix_chunk_text_locale", "chunk", ["text_id", "locale_code"], schema="rag")
    op.create_index("ix_chunk_kind_locale", "chunk", ["kind", "locale_code"], schema="rag")
    op.execute(
        "CREATE INDEX ix_chunk_embedding_hnsw ON rag.chunk "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )

    op.create_table(
        "embedding_cache",
        sa.Column("text_sha256", sa.String(64), primary_key=True),
        sa.Column("embedding_model", sa.String(128), primary_key=True),
        sa.Column("embedding", Vector(DIM), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        schema="rag",
    )

    op.create_table(
        "sync_state",
        sa.Column("key", sa.String(64), primary_key=True),
        sa.Column("value", pg.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        schema="rag",
    )

    op.create_table(
        "translation_audit",
        sa.Column("audit_id", sa.BigInteger, sa.Identity(), primary_key=True),
        sa.Column(
            "translation_id",
            sa.BigInteger,
            sa.ForeignKey("i18n_translation.translation_id", ondelete="SET NULL"),
        ),
        sa.Column("text_id", sa.BigInteger, nullable=False),
        sa.Column("locale_code", sa.String(16), nullable=False),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("prompt_sha256", sa.String(64), nullable=False),
        sa.Column(
            "tm_chunk_ids",
            pg.ARRAY(sa.BigInteger),
            nullable=False,
            server_default=sa.text("'{}'::bigint[]"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        schema="rag",
    )

    op.create_table(
        "review_queue",
        sa.Column("review_id", sa.BigInteger, sa.Identity(), primary_key=True),
        sa.Column("text_id", sa.BigInteger, nullable=False),
        sa.Column("locale_code", sa.String(16), nullable=False),
        sa.Column("source_text", sa.Text, nullable=False),
        sa.Column("candidate_text", sa.Text),
        sa.Column("reason", sa.Text, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        schema="rag",
    )


def downgrade() -> None:
    for table in ("review_queue", "translation_audit", "sync_state", "embedding_cache", "chunk"):
        op.drop_table(table, schema="rag")
    op.drop_table("document", schema="rag")
    op.execute("DROP SCHEMA IF EXISTS rag")
