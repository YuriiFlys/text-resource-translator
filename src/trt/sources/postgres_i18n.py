"""S1: інкрементальне витягування рядків з i18n_* у JSONL-знімок на диску."""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from trt.config import get_settings
from trt.logging import get_logger
from trt.models import I18nLocale, I18nText, I18nTranslation
from trt.sources.base import FetchResult, get_state, iso, register_artifact, set_state
from trt.storage.raw_store import RawStore

log = get_logger(__name__)

SOURCE = "postgres"
STATE_KEY = "fetch.postgres"


def enabled_locales(session: Session) -> list[str]:
    return list(session.scalars(select(I18nLocale.locale_code).where(I18nLocale.enabled)).all())


def fetch_i18n(session: Session, store: RawStore, since: datetime | None = None) -> FetchResult:
    """Вибрати переклади, оновлені після курсора, і зберегти як JSONL у RawStore."""
    state = get_state(session, STATE_KEY)
    if since is None and state.get("since"):
        since = datetime.fromisoformat(state["since"])

    stmt = (
        select(
            I18nText.text_id,
            I18nText.text_code,
            I18nText.owner_kind,
            I18nTranslation.translation_id,
            I18nTranslation.locale_code,
            I18nTranslation.translated_text,
            I18nTranslation.date_updated,
        )
        .join(I18nTranslation, I18nTranslation.text_id == I18nText.text_id)
        .order_by(I18nTranslation.date_updated, I18nTranslation.translation_id)
    )
    if since is not None:
        stmt = stmt.where(I18nTranslation.date_updated > since)

    rows = session.execute(stmt).all()
    result = FetchResult(source=SOURCE)
    if not rows:
        log.info("fetch.postgres.nothing_new", since=iso(since))
        return result

    lines = [
        json.dumps(
            {
                "text_id": r.text_id,
                "text_code": r.text_code,
                "owner_kind": r.owner_kind,
                "translation_id": r.translation_id,
                "locale_code": r.locale_code,
                "translated_text": r.translated_text,
                "date_updated": iso(r.date_updated),
            },
            ensure_ascii=False,
        )
        for r in rows
    ]
    data = ("\n".join(lines) + "\n").encode("utf-8")
    max_updated: datetime = max(r.date_updated for r in rows)
    snapshot_name = f"i18n-{max_updated.strftime('%Y%m%dT%H%M%S')}"
    stored = store.put(
        SOURCE,
        data,
        "jsonl",
        meta={
            "name": snapshot_name,
            "rows": len(rows),
            "since": iso(since),
            "max_date_updated": iso(max_updated),
            "dsn_host": get_settings().pg_dsn.split("@")[-1],
        },
    )
    artifact = register_artifact(
        session,
        source=SOURCE,
        name=snapshot_name,
        path=str(stored.path),
        sha256=stored.sha256,
        size_bytes=stored.size_bytes,
        meta={"rows": len(rows), "since": iso(since), "max_date_updated": iso(max_updated)},
    )
    if artifact is None:
        result.skipped += 1
    else:
        result.artifacts.append(artifact)

    set_state(session, STATE_KEY, {"since": iso(max_updated), "last_rows": len(rows)})
    log.info("fetch.postgres.done", rows=len(rows), path=str(stored.path), since=iso(since))
    return result
