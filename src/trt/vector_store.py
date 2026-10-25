"""rag.chunk: запис фрагментів із векторами та kNN-пошук через pgvector."""

from __future__ import annotations

from dataclasses import dataclass

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, delete, select, text
from sqlalchemy.orm import Session

from trt.chunking import embed_text_for, split_document
from trt.config import get_settings
from trt.embeddings import CachedEmbedder, get_embeddings
from trt.logging import get_logger
from trt.models import RagChunk, RagDocument

log = get_logger(__name__)


@dataclass
class EmbedStats:
    documents: int = 0
    chunks: int = 0


@dataclass
class Hit:
    chunk_id: int
    document_id: int
    chunk_index: int
    text: str
    doc_key: str
    text_id: int | None
    locale_code: str | None
    kind: str
    score: float


def make_embedder(session: Session) -> CachedEmbedder:
    s = get_settings()
    return CachedEmbedder(session, get_embeddings(), s.embedding_model, s.embedding_dim)


def embed_pending(
    session: Session, *, force_all: bool = False, batch_docs: int = 200
) -> EmbedStats:
    """Чанкувати та векторизувати документи, у яких змінився вміст (або всі при force_all)."""
    embedder = make_embedder(session)
    stats = EmbedStats()
    stmt = select(RagDocument).order_by(RagDocument.document_id)
    if not force_all:
        stmt = stmt.where(
            (RagDocument.chunked_sha256.is_(None))
            | (RagDocument.chunked_sha256 != RagDocument.content_sha256)
        )
    docs = session.scalars(stmt).all()

    for start in range(0, len(docs), batch_docs):
        batch = docs[start : start + batch_docs]
        pieces = [(doc, lc_doc) for doc in batch for lc_doc in split_document(doc)]
        if not pieces:
            for doc in batch:
                doc.chunked_sha256 = doc.content_sha256
            continue
        vectors = embedder.embed_documents(
            [embed_text_for(doc, lc.page_content) for doc, lc in pieces]
        )
        session.execute(
            delete(RagChunk).where(RagChunk.document_id.in_([d.document_id for d in batch]))
        )
        session.add_all(
            RagChunk(
                document_id=doc.document_id,
                chunk_index=lc.metadata["chunk_index"],
                text=lc.page_content,
                embed_text=embed_text_for(doc, lc.page_content),
                embedding=vec,
                embedding_model=embedder.model_name,
                kind=doc.kind,
                locale_code=doc.locale_code,
                text_id=doc.text_id,
                owner_kind=doc.owner_kind,
            )
            for (doc, lc), vec in zip(pieces, vectors, strict=True)
        )
        for doc in batch:
            doc.chunked_sha256 = doc.content_sha256
        session.flush()
        stats.documents += len(batch)
        stats.chunks += len(pieces)
        log.info("embed.batch", documents=len(batch), chunks=len(pieces))
    return stats


SEARCH_SQL = text(
    """
    SELECT c.chunk_id, c.document_id, c.chunk_index, c.text, d.doc_key,
           c.text_id, c.locale_code, c.kind,
           1 - (c.embedding <=> :q) AS score
    FROM rag.chunk c
    JOIN rag.document d USING (document_id)
    WHERE (CAST(:locale AS varchar) IS NULL OR c.locale_code = :locale)
      AND (CAST(:kind AS varchar) IS NULL OR c.kind = :kind)
      AND (CAST(:exclude_text_id AS bigint) IS NULL OR c.text_id IS DISTINCT FROM :exclude_text_id)
    ORDER BY c.embedding <=> :q
    LIMIT :n
    """
)


def search_similar(
    session: Session,
    query_vector: list[float],
    n: int = 5,
    *,
    locale: str | None = None,
    kind: str | None = None,
    exclude_text_id: int | None = None,
) -> list[Hit]:
    """N найближчих фрагментів до вектора; повертає текст, chunk_id, document_id і score."""
    dim = get_settings().embedding_dim
    stmt = SEARCH_SQL.bindparams(bindparam("q", type_=Vector(dim)))
    session.execute(
        text("SELECT set_config('hnsw.ef_search', :ef, true)"), {"ef": str(max(40, n * 10))}
    )
    rows = session.execute(
        stmt,
        {
            "q": query_vector,
            "n": n,
            "locale": locale,
            "kind": kind,
            "exclude_text_id": exclude_text_id,
        },
    ).all()
    return [Hit(**row._mapping) for row in rows]


def search_text(session: Session, query: str, n: int = 5, **filters) -> list[Hit]:
    """Те саме, але за текстом: вектор запиту рахується тією ж моделлю."""
    vector = make_embedder(session).embed_query(query)
    return search_similar(session, vector, n, **filters)
