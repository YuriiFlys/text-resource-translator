from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from trt.config import get_settings


@lru_cache
def get_engine() -> Engine:
    return create_engine(get_settings().pg_dsn, pool_pre_ping=True, future=True)


@lru_cache
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Одна транзакція на одиницю роботи: commit при успіху, rollback при помилці."""
    session = _session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ping() -> str:
    with get_engine().connect() as conn:
        return conn.execute(text("select version()")).scalar_one()


def has_pgvector() -> bool:
    with get_engine().connect() as conn:
        row = conn.execute(
            text("select 1 from pg_available_extensions where name = 'vector'")
        ).first()
        return row is not None
