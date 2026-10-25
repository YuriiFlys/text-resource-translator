from __future__ import annotations

import os
from pathlib import Path

import pytest

TEST_PG_DSN = os.environ.get("TEST_PG_DSN")

requires_pg = pytest.mark.skipif(
    not TEST_PG_DSN, reason="set TEST_PG_DSN to run integration tests against PostgreSQL+pgvector"
)

ROOT = Path(__file__).resolve().parents[1]


def reset_caches() -> None:
    """Скинути lru_cache налаштувань/engine/embeddings після зміни оточення."""
    from trt import config, db, embeddings

    config.get_settings.cache_clear()
    db.get_engine.cache_clear()
    db._session_factory.cache_clear()
    embeddings.get_embeddings.cache_clear()


@pytest.fixture(scope="session")
def pg_env(tmp_path_factory):
    """Чиста схема (downgrade→upgrade) + seed; оточення для hashing-вкладень."""
    if not TEST_PG_DSN:
        pytest.skip("TEST_PG_DSN not set")
    raw_dir = tmp_path_factory.mktemp("raw")
    env = {
        "PG_DSN": TEST_PG_DSN,
        "EMBEDDING_PROVIDER": "hashing",
        "EMBEDDING_MODEL": "hashing-test",
        "EMBEDDING_DIM": "768",
        "RAW_DIR": str(raw_dir),
        "SOURCES_FILE": str(ROOT / "sources.yaml"),
        "SOURCE_LOCALE": "en",
        "LLM_MODEL": "fake",
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    reset_caches()

    from alembic import command
    from sqlalchemy import text

    from trt.cli import _alembic_config
    from trt.db import get_engine

    cfg = _alembic_config()
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    with get_engine().begin() as conn:
        conn.execute(text((ROOT / "sql" / "seed_i18n.sql").read_text(encoding="utf-8")))
    yield env
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    reset_caches()
