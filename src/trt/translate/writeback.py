"""Запис перекладів у i18n_translation (upsert) + аудит + черга на перевірку."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session

from trt.models import RagReviewQueue, RagTranslationAudit

UPSERT_SQL = text(
    """
    INSERT INTO i18n_translation (text_id, locale_code, translated_text, date_updated)
    VALUES (:text_id, :locale_code, :translated_text, now())
    ON CONFLICT (text_id, locale_code) DO UPDATE
       SET translated_text = EXCLUDED.translated_text,
           date_updated = now()
    RETURNING translation_id
    """
)


def write_translation(
    session: Session,
    *,
    text_id: int,
    locale_code: str,
    translated_text: str,
    model: str,
    prompt_sha256: str,
    tm_chunk_ids: list[int],
) -> int:
    translation_id = session.execute(
        UPSERT_SQL,
        {"text_id": text_id, "locale_code": locale_code, "translated_text": translated_text},
    ).scalar_one()
    session.add(
        RagTranslationAudit(
            translation_id=translation_id,
            text_id=text_id,
            locale_code=locale_code,
            model=model,
            prompt_sha256=prompt_sha256,
            tm_chunk_ids=tm_chunk_ids,
        )
    )
    return translation_id


def queue_for_review(
    session: Session,
    *,
    text_id: int,
    locale_code: str,
    source_text: str,
    candidate_text: str | None,
    reason: str,
) -> None:
    session.add(
        RagReviewQueue(
            text_id=text_id,
            locale_code=locale_code,
            source_text=source_text,
            candidate_text=candidate_text,
            reason=reason,
        )
    )
