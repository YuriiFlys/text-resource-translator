from __future__ import annotations

from sqlalchemy import create_engine, inspect, text

from tests.conftest import requires_pg


@requires_pg
def test_schema_objects(pg_env):
    engine = create_engine(pg_env["PG_DSN"])
    insp = inspect(engine)
    assert {"i18n_locale", "i18n_text", "i18n_translation"} <= set(insp.get_table_names())
    assert {
        "document",
        "chunk",
        "embedding_cache",
        "sync_state",
        "translation_audit",
        "review_queue",
        "raw_artifact",
    } <= set(insp.get_table_names(schema="rag"))
    idx = {i["name"] for i in insp.get_indexes("chunk", schema="rag")}
    assert "ix_chunk_embedding_hnsw" in idx
    engine.dispose()


@requires_pg
def test_vector_knn_order(pg_env):
    dim = int(pg_env["EMBEDDING_DIM"])
    engine = create_engine(pg_env["PG_DSN"])
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO rag.document (doc_key, source, kind, content, content_sha256) "
                "VALUES ('t:knn', 'test', 'ui_string', 'a', repeat('0', 64)) "
                "ON CONFLICT (doc_key) DO NOTHING"
            )
        )
        doc_id = conn.execute(
            text("SELECT document_id FROM rag.document WHERE doc_key='t:knn'")
        ).scalar_one()
        conn.execute(text("DELETE FROM rag.chunk WHERE document_id = :d"), {"d": doc_id})
        vecs = ([1.0] + [0.0] * (dim - 1), [0.0, 1.0] + [0.0] * (dim - 2))
        for i, v in enumerate(vecs):
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
        conn.execute(text("DELETE FROM rag.document WHERE doc_key='t:knn'"))
    engine.dispose()
    assert rows == ["chunk0", "chunk1"]
