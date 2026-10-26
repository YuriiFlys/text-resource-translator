"""S2/S3: завантаження файлів ресурсів та довідкових документів за sources.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import httpx
import yaml
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from trt.logging import get_logger
from trt.sources.base import FetchResult, get_state, register_artifact, set_state
from trt.storage.raw_store import RawStore

log = get_logger(__name__)

SOURCE_FILES = "resource_file"
SOURCE_DOCS = "document"


class SourceEntry(BaseModel):
    name: str
    type: Literal["resource_file", "document"]
    location: str = Field(description="http(s) URL або локальний шлях")
    format: str
    locale: str | None = None
    owner_kind: str | None = None
    description: str | None = None


class SourcesConfig(BaseModel):
    sources: list[SourceEntry] = Field(default_factory=list)


def load_sources(path: Path) -> SourcesConfig:
    if not path.exists():
        return SourcesConfig()
    return SourcesConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _download(entry: SourceEntry, state: dict) -> tuple[bytes | None, dict]:
    """Повертає (bytes або None якщо не змінилось, оновлений state з ETag/Last-Modified)."""
    if entry.location.startswith(("http://", "https://")):
        headers = {}
        if state.get("etag"):
            headers["If-None-Match"] = state["etag"]
        if state.get("last_modified"):
            headers["If-Modified-Since"] = state["last_modified"]
        with httpx.Client(follow_redirects=True, timeout=60) as client:
            resp = client.get(entry.location, headers=headers)
        if resp.status_code == 304:
            return None, state
        resp.raise_for_status()
        new_state = {
            "etag": resp.headers.get("ETag"),
            "last_modified": resp.headers.get("Last-Modified"),
        }
        return resp.content, {k: v for k, v in new_state.items() if v}
    return Path(entry.location).read_bytes(), state


def fetch_files(
    session: Session, store: RawStore, config: SourcesConfig, only_type: str | None = None
) -> list[FetchResult]:
    results: dict[str, FetchResult] = {
        SOURCE_FILES: FetchResult(SOURCE_FILES),
        SOURCE_DOCS: FetchResult(SOURCE_DOCS),
    }
    for entry in config.sources:
        if only_type and entry.type != only_type:
            continue
        state_key = f"fetch.file.{entry.name}"
        state = get_state(session, state_key)
        try:
            data, new_state = _download(entry, state)
        except Exception as exc:  # noqa: BLE001 — одна невдала адреса не зупиняє решту
            log.warning("fetch.file.failed", name=entry.name, error=str(exc))
            continue
        result = results[entry.type]
        if data is None:
            result.skipped += 1
            log.info("fetch.file.not_modified", name=entry.name)
            continue
        stored = store.put(
            entry.type,
            data,
            entry.format,
            meta={"name": entry.name, "location": entry.location, **entry.model_dump()},
        )
        artifact = register_artifact(
            session,
            source=entry.type,
            name=entry.name,
            path=str(stored.path),
            sha256=stored.sha256,
            size_bytes=stored.size_bytes,
            meta=entry.model_dump(),
        )
        if artifact is None:
            result.skipped += 1
            log.info("fetch.file.unchanged", name=entry.name, sha256=stored.sha256[:12])
        else:
            result.artifacts.append(artifact)
            log.info("fetch.file.done", name=entry.name, bytes=stored.size_bytes)
        set_state(session, state_key, {**new_state, "sha256": stored.sha256})
    return [r for r in results.values() if r.artifacts or r.skipped]
