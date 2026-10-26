"""Оркестрація кроків: fetch → ingest → embed → translate."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import text

from trt.config import get_settings
from trt.db import session_scope
from trt.ingest.documents import IngestStats, ingest_pending
from trt.logging import get_logger
from trt.sources.base import FetchResult
from trt.sources.postgres_i18n import enabled_locales, fetch_i18n
from trt.sources.resource_files import fetch_files, load_sources
from trt.storage.raw_store import RawStore
from trt.vector_store import EmbedStats, embed_pending

log = get_logger(__name__)

RUN_ALL_LOCK = "trt.run_all"


@dataclass
class RunReport:
    fetched: list[FetchResult] = field(default_factory=list)
    ingest: IngestStats | None = None
    embed: EmbedStats | None = None
    translated: list = field(default_factory=list)
    skipped_locked: bool = False


def raw_store() -> RawStore:
    return RawStore(get_settings().raw_dir)


def step_fetch(source: str = "all", since: datetime | None = None) -> list[FetchResult]:
    store = raw_store()
    results: list[FetchResult] = []
    with session_scope() as session:
        if source in ("pg", "all"):
            results.append(fetch_i18n(session, store, since))
        if source in ("files", "docs", "all"):
            cfg = load_sources(get_settings().sources_file)
            only = {"files": "resource_file", "docs": "document"}.get(source)
            results.extend(fetch_files(session, store, cfg, only_type=only))
    return results


def step_ingest() -> IngestStats:
    with session_scope() as session:
        return ingest_pending(session, raw_store())


def step_embed(force_all: bool = False) -> EmbedStats:
    with session_scope() as session:
        return embed_pending(session, force_all=force_all)


def step_translate(locales: list[str] | None, limit: int, dry_run: bool, llm=None) -> list:
    from trt.translate.translator import Translator

    translator = Translator(llm)
    reports = []
    with session_scope() as session:
        targets = locales or [
            lc for lc in enabled_locales(session) if lc != get_settings().source_locale
        ]
        for locale in targets:
            reports.append(
                translator.translate_locale(session, locale, limit=limit, dry_run=dry_run)
            )
    return reports


def run_all(
    *, locales: list[str] | None = None, limit: int = 100, dry_run: bool = False, llm=None
) -> RunReport:
    """Повний цикл. Захищений advisory lock'ом від паралельного запуску."""
    report = RunReport()
    with session_scope() as lock_session:
        locked = lock_session.execute(
            text("SELECT pg_try_advisory_lock(hashtext(:k))"), {"k": RUN_ALL_LOCK}
        ).scalar_one()
        if not locked:
            log.warning("run_all.already_running")
            report.skipped_locked = True
            return report
        try:
            report.fetched = step_fetch("all")
            report.ingest = step_ingest()
            report.embed = step_embed()
            report.translated = step_translate(locales, limit, dry_run, llm)
            # нові переклади → у документи/вектори, щоб стати translation memory
            report.fetched += step_fetch("pg")
            step_ingest()
            step_embed()
        finally:
            lock_session.execute(
                text("SELECT pg_advisory_unlock(hashtext(:k))"), {"k": RUN_ALL_LOCK}
            )
    return report
