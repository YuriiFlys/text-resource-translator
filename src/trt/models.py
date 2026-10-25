"""ORM-моделі: продуктові таблиці i18n_* (public) та RAG-шар (schema rag)."""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from trt.config import get_settings

EMBEDDING_DIM = get_settings().embedding_dim


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# public.i18n_* — схема із завдання, джерело істини для перекладів
# ---------------------------------------------------------------------------


class I18nLocale(Base):
    __tablename__ = "i18n_locale"

    locale_code: Mapped[str] = mapped_column(String(16), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(64), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")


class I18nText(Base):
    __tablename__ = "i18n_text"

    text_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    text_code: Mapped[str | None] = mapped_column(String(256), unique=True)
    owner_kind: Mapped[str | None] = mapped_column(String(20))

    translations: Mapped[list[I18nTranslation]] = relationship(
        back_populates="text", cascade="all, delete-orphan"
    )


class I18nTranslation(Base):
    __tablename__ = "i18n_translation"
    __table_args__ = (
        UniqueConstraint("text_id", "locale_code", name="uq_i18n_translation_text_locale"),
    )

    translation_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    text_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("i18n_text.text_id", ondelete="CASCADE"), nullable=False
    )
    locale_code: Mapped[str] = mapped_column(
        String(16), ForeignKey("i18n_locale.locale_code", ondelete="CASCADE"), nullable=False
    )
    translated_text: Mapped[str] = mapped_column(Text, nullable=False)
    date_updated: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    text: Mapped[I18nText] = relationship(back_populates="translations")


# ---------------------------------------------------------------------------
# rag.* — повні документи, фрагменти з векторами, службові таблиці конвеєра
# ---------------------------------------------------------------------------

RAG_SCHEMA = "rag"


class RagDocument(Base):
    __tablename__ = "document"
    __table_args__ = (
        Index("ix_document_text_locale", "text_id", "locale_code"),
        Index("ix_document_text_code", "text_code"),
        Index("ix_document_sha", "content_sha256"),
        Index("ix_document_source_upd", "source", "source_updated_at"),
        CheckConstraint("kind in ('ui_string', 'reference_doc')", name="ck_document_kind"),
        {"schema": RAG_SCHEMA},
    )

    document_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    doc_key: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    text_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("i18n_text.text_id", ondelete="SET NULL")
    )
    text_code: Mapped[str | None] = mapped_column(String(256))
    owner_kind: Mapped[str | None] = mapped_column(String(20))
    locale_code: Mapped[str | None] = mapped_column(
        String(16), ForeignKey("i18n_locale.locale_code")
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    pages: Mapped[list | None] = mapped_column(JSONB)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    raw_path: Mapped[str | None] = mapped_column(Text)
    source_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    chunked_sha256: Mapped[str | None] = mapped_column(String(64))

    chunks: Mapped[list[RagChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class RagChunk(Base):
    __tablename__ = "chunk"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_document_index"),
        Index("ix_chunk_text_locale", "text_id", "locale_code"),
        Index("ix_chunk_kind_locale", "kind", "locale_code"),
        Index(
            "ix_chunk_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        {"schema": RAG_SCHEMA},
    )

    chunk_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(f"{RAG_SCHEMA}.document.document_id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embed_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    locale_code: Mapped[str | None] = mapped_column(String(16))
    text_id: Mapped[int | None] = mapped_column(BigInteger)
    owner_kind: Mapped[str | None] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped[RagDocument] = relationship(back_populates="chunks")


class RagEmbeddingCache(Base):
    __tablename__ = "embedding_cache"
    __table_args__ = {"schema": RAG_SCHEMA}

    text_sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    embedding_model: Mapped[str] = mapped_column(String(128), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class RagSyncState(Base):
    """Курсори інкрементальної синхронізації та час останніх запусків."""

    __tablename__ = "sync_state"
    __table_args__ = {"schema": RAG_SCHEMA}

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class RagTranslationAudit(Base):
    __tablename__ = "translation_audit"
    __table_args__ = {"schema": RAG_SCHEMA}

    audit_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    translation_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("i18n_translation.translation_id", ondelete="SET NULL")
    )
    text_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    locale_code: Mapped[str] = mapped_column(String(16), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    tm_chunk_ids: Mapped[list[int]] = mapped_column(ARRAY(BigInteger), nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class RagReviewQueue(Base):
    """Переклади, що не пройшли валідацію і чекають ручної перевірки."""

    __tablename__ = "review_queue"
    __table_args__ = {"schema": RAG_SCHEMA}

    review_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    text_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    locale_code: Mapped[str] = mapped_column(String(16), nullable=False)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    candidate_text: Mapped[str | None] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
