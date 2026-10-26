from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from trt.models import RagRawArtifact, RagSyncState


@dataclass
class FetchResult:
    source: str
    artifacts: list[RagRawArtifact] = field(default_factory=list)
    skipped: int = 0  # файли без змін (той самий sha256)


def register_artifact(
    session: Session, *, source: str, name: str, path: str, sha256: str, size_bytes: int, meta: dict
) -> RagRawArtifact | None:
    """Записати артефакт у rag.raw_artifact; None, якщо такий (source,name,sha) уже є."""
    existing = session.scalar(
        select(RagRawArtifact).where(
            RagRawArtifact.source == source,
            RagRawArtifact.name == name,
            RagRawArtifact.sha256 == sha256,
        )
    )
    if existing is not None:
        return None
    artifact = RagRawArtifact(
        source=source, name=name, path=path, sha256=sha256, size_bytes=size_bytes, meta=meta
    )
    session.add(artifact)
    session.flush()
    return artifact


def get_state(session: Session, key: str) -> dict:
    row = session.get(RagSyncState, key)
    return dict(row.value) if row else {}


def set_state(session: Session, key: str, value: dict) -> None:
    row = session.get(RagSyncState, key)
    if row is None:
        session.add(RagSyncState(key=key, value=value))
    else:
        row.value = value
    session.flush()


def iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None
