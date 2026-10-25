from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, inspect, text

from tests.conftest import TEST_PG_DSN, requires_pg


@pytest.fixture(scope="module")
def migrated_engine():
    os.environ["PG_DSN"] = TEST_PG_DSN
    from alembic import command
    from alembic.config import Config

    from trt.cli import _alembic_config

    cfg: Config = _alembic_config()
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    engine = create_engine(TEST_PG_DSN)
    yield engine
    engine.dispose()


@requires_pg
def test_schema_objects(migrated_engine):
    insp = inspect(migrated_engine)
    assert {"i18n_locale", "i18n_text", "i18n_translation"} <= set(insp.get_table_names())
    assert {
        "document",
        "chunk",
        "embedding_cache",
        "sync_state",
        "translation_audit",
        "review_queue",
    } <= set(insp.get_table_names(schema="rag"))
    idx = {i["name"] for i in insp.get_indexes("chunk", schema="rag")}
    assert "ix_chunk_embedding_hnsw" in idx


@requires_pg
def test_vector_roundtrip_and_knn(migrated_engine):
    dim = int(os.environ.get("EMBEDDING_DIM", "768"))
    with migrated_engine.begin() as conn:
        conn.execute(text(open("sql/seed_i18n.sql", encoding="utf-8").read()))
        conn.execute(
            text(
                "INSERT INTO rag.document (doc_key, source, kind, content, content_sha256) "
                "VALUES ('t:1', 'test', 'ui_string', 'a', repeat('0', 64)) "
                "ON CONFLICT (doc_key) DO NOTHING"
            )
        )
        doc_id = conn.execute(
            text("SELECT document_id FROM rag.document WHERE doc_key='t:1'")
        ).scalar_one()
        conn.execute(text("DELETE FROM rag.chunk WHERE document_id = :d"), {"d": doc_id})
        for i, v in enumerate(([1.0] + [0.0] * (dim - 1), [0.0, 1.0] + [0.0] * (dim - 2))):
            conn.execute(
                text(
                    "INSERT INTO rag.chunk (document_id, chunk_index, text, embed_text, embedding, "
                    "embedding_model, kind) VALUES (:d, :i, :t, :t, :e, 'test', 'ui_string')"
                ),
                {"d": doc_id, "i": i, "t": f"chunk{i}", "e": str(v)},
            )
        q = str([0.9, 0.1] + [0.0] * (dim - 2))
        rows = (
            conn.execute(
                text(
                    "SELECT text FROM rag.chunk WHERE document_id=:d "
                    "ORDER BY embedding <=> :q LIMIT 2"
                ),
                {"d": doc_id, "q": q},
            )
            .scalars()
            .all()
        )
    assert rows == ["chunk0", "chunk1"]
