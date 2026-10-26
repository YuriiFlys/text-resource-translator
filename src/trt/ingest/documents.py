"""Ingest: сирі артефакти → повні документи в rag.document (upsert за doc_key)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from trt.extract.document_text import extract_document
from trt.extract.resource_parsers import parse_resource
from trt.logging import get_logger
from trt.models import RagDocument, RagRawArtifact
from trt.storage.raw_store import RawStore

log = get_logger(__name__)


@dataclass
class IngestStats:
    artifacts: int = 0
    upserted: int = 0
    unchanged: int = 0
    errors: list[str] = field(default_factory=list)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def upsert_document(session: Session, doc: dict) -> bool:
    """INSERT ... ON CONFLICT (doc_key): оновлює лише коли змінився вміст. True = запис змінено."""
    doc = {**doc, "content_sha256": sha256_text(doc["content"])}
    stmt = insert(RagDocument).values(**doc)
    update_cols = {k: getattr(stmt.excluded, k) for k in doc if k not in ("doc_key",)}
    update_cols["ingested_at"] = datetime.now(UTC)
    stmt = stmt.on_conflict_do_update(
        index_elements=[RagDocument.doc_key],
        set_=update_cols,
        where=RagDocument.content_sha256 != stmt.excluded.content_sha256,
    ).returning(RagDocument.document_id)
    return session.execute(stmt).first() is not None


def _ingest_postgres_snapshot(session: Session, data: bytes, raw_path: str) -> tuple[int, int]:
    changed = unchanged = 0
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("translated_text", "").strip():
            continue
        doc = {
            "doc_key": f"pg:text:{row['text_id']}:{row['locale_code']}",
            "source": "postgres",
            "kind": "ui_string",
            "text_id": row["text_id"],
            "text_code": row.get("text_code"),
            "owner_kind": row.get("owner_kind"),
            "locale_code": row["locale_code"],
            "content": row["translated_text"],
            "raw_path": raw_path,
            "source_updated_at": (
                datetime.fromisoformat(row["date_updated"]) if row.get("date_updated") else None
            ),
        }
        if upsert_document(session, doc):
            changed += 1
        else:
            unchanged += 1
    return changed, unchanged


def _ingest_resource_file(
    session: Session, artifact: RagRawArtifact, data: bytes
) -> tuple[int, int]:
    meta = artifact.meta
    entries = parse_resource(meta["format"], data)
    locale = meta.get("locale")
    changed = unchanged = 0
    for e in entries:
        doc = {
            "doc_key": f"file:{artifact.name}:{e.key}:{locale or '-'}",
            "source": "resource_file",
            "kind": "ui_string",
            "text_code": e.key[:256],
            "owner_kind": (meta.get("owner_kind") or None),
            "locale_code": locale,
            "content": e.value,
            "raw_path": artifact.path,
            "pages": {"comment": e.comment} if e.comment else None,
        }
        if upsert_document(session, doc):
            changed += 1
        else:
            unchanged += 1
    return changed, unchanged


def _ingest_reference_document(
    session: Session, artifact: RagRawArtifact, data: bytes
) -> tuple[int, int]:
    meta = artifact.meta
    extracted = extract_document(meta["format"], data)
    if not extracted.text.strip():
        raise ValueError(f"no text extracted from {artifact.path}")
    doc = {
        "doc_key": f"doc:{artifact.name}",
        "source": "document",
        "kind": "reference_doc",
        "text_code": artifact.name[:256],
        "locale_code": meta.get("locale"),
        "content": extracted.text,
        "pages": extracted.pages,
        "raw_path": artifact.path,
    }
    return (1, 0) if upsert_document(session, doc) else (0, 1)


HANDLERS = {
    "postgres": lambda s, a, d: _ingest_postgres_snapshot(s, d, a.path),
    "resource_file": _ingest_resource_file,
    "document": _ingest_reference_document,
}


def ingest_pending(session: Session, store: RawStore) -> IngestStats:
    """Обробити всі артефакти з ingested_at IS NULL у порядку надходження."""
    stats = IngestStats()
    pending = session.scalars(
        select(RagRawArtifact)
        .where(RagRawArtifact.ingested_at.is_(None))
        .order_by(RagRawArtifact.fetched_at, RagRawArtifact.raw_id)
    ).all()
    for artifact in pending:
        stats.artifacts += 1
        try:
            data = store.read(Path(artifact.path))
            changed, unchanged = HANDLERS[artifact.source](session, artifact, data)
        except Exception as exc:  # noqa: BLE001 — помилка одного файлу не блокує інші
            msg = f"{artifact.source}/{artifact.name}: {exc}"
            stats.errors.append(msg)
            log.error("ingest.failed", artifact=artifact.name, error=str(exc))
            continue
        artifact.ingested_at = datetime.now(UTC)
        stats.upserted += changed
        stats.unchanged += unchanged
        log.info("ingest.done", artifact=artifact.name, changed=changed, unchanged=unchanged)
    session.flush()
    return stats
