"""LLM-переклад з контекстом із векторного пошуку (translation memory + глосарій)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from trt.config import get_settings
from trt.logging import get_logger
from trt.models import I18nText, I18nTranslation
from trt.translate.prompt import (
    TMExample,
    TranslationBatch,
    TranslationContext,
    WorkItem,
    build_messages,
)
from trt.translate.validation import validate_translation
from trt.translate.writeback import queue_for_review, write_translation
from trt.vector_store import make_embedder, search_similar

log = get_logger(__name__)

WORK_SQL = text(
    """
    SELECT t.text_id, t.text_code, t.owner_kind, src.translated_text AS source_text
    FROM i18n_text t
    JOIN i18n_translation src
      ON src.text_id = t.text_id AND src.locale_code = :source_locale
    LEFT JOIN i18n_translation tgt
      ON tgt.text_id = t.text_id AND tgt.locale_code = :target_locale
    WHERE tgt.translation_id IS NULL OR tgt.date_updated < src.date_updated
    ORDER BY t.text_id
    LIMIT :limit
    """
)


@dataclass
class TranslateStats:
    locale: str
    requested: int = 0
    written: int = 0
    queued_for_review: int = 0
    llm_calls: int = 0
    results: list[tuple[str, str, str]] = field(default_factory=list)  # (code, source, target)


def select_work(
    session: Session, source_locale: str, target_locale: str, limit: int
) -> list[WorkItem]:
    rows = session.execute(
        WORK_SQL, {"source_locale": source_locale, "target_locale": target_locale, "limit": limit}
    ).all()
    return [WorkItem(r.text_id, r.text_code, r.owner_kind, r.source_text) for r in rows]


def build_context(
    session: Session,
    item: WorkItem,
    *,
    source_locale: str,
    target_locale: str,
    k_tm: int = 5,
    k_glossary: int = 2,
) -> TranslationContext:
    """kNN серед UI-рядків вихідної локалі → їхні наявні переклади = translation memory."""
    embedder = make_embedder(session)
    prefix = " | ".join(p for p in (item.text_code, item.owner_kind) if p)
    query_vec = embedder.embed_query(
        f"{prefix}: {item.source_text}" if prefix else item.source_text
    )

    ctx = TranslationContext()
    tm_hits = search_similar(
        session,
        query_vec,
        n=k_tm * 3,
        locale=source_locale,
        kind="ui_string",
        exclude_text_id=item.text_id,
    )
    text_ids = [h.text_id for h in tm_hits if h.text_id is not None]
    if text_ids:
        rows = session.execute(
            select(I18nTranslation.text_id, I18nTranslation.translated_text, I18nText.text_code)
            .join(I18nText, I18nText.text_id == I18nTranslation.text_id)
            .where(
                I18nTranslation.locale_code == target_locale,
                I18nTranslation.text_id.in_(text_ids),
            )
        ).all()
        targets = {r.text_id: (r.translated_text, r.text_code) for r in rows}
        for h in tm_hits:
            if h.text_id in targets and len(ctx.tm_examples) < k_tm:
                tgt, code = targets[h.text_id]
                ctx.tm_examples.append(TMExample(h.chunk_id, h.text, tgt, code, h.score))

    for h in search_similar(session, query_vec, n=k_glossary, kind="reference_doc"):
        ctx.glossary_snippets.append((h.chunk_id, h.text))
    return ctx


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_batch(raw: str) -> TranslationBatch:
    cleaned = _FENCE.sub("", raw.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("LLM answer contains no JSON object")
    return TranslationBatch.model_validate_json(cleaned[start : end + 1])


def get_chat_model(model: str | None = None) -> BaseChatModel:
    from langchain.chat_models import init_chat_model

    # Провайдер виводиться з назви моделі (claude-* → anthropic, gpt-* → openai, ...).
    return init_chat_model(model or get_settings().llm_model)


class Translator:
    def __init__(
        self, llm: BaseChatModel | None = None, *, batch_size: int = 20, max_retries: int = 1
    ):
        self._llm = llm
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.model_name = get_settings().llm_model

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = get_chat_model()
        return self._llm

    def _call(self, messages: list[tuple[str, str]]) -> tuple[dict[str, str], str]:
        response = self.llm.invoke(messages)
        content = (
            response.content if isinstance(response.content, str) else json.dumps(response.content)
        )
        batch = parse_batch(content)
        prompt_sha = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()
        return {it.text_code: it.translation for it in batch.items}, prompt_sha

    def translate_locale(
        self, session: Session, target_locale: str, *, limit: int = 100, dry_run: bool = False
    ) -> TranslateStats:
        settings = get_settings()
        source_locale = settings.source_locale
        stats = TranslateStats(locale=target_locale)
        items = select_work(session, source_locale, target_locale, limit)
        stats.requested = len(items)
        if not items:
            return stats

        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            by_code = {it.text_code or f"text_id:{it.text_id}": it for it in batch}
            contexts = {
                it.text_id: build_context(
                    session, it, source_locale=source_locale, target_locale=target_locale
                )
                for it in batch
            }
            pending = dict(by_code)
            errors: dict[str, str] = {}
            accepted: dict[str, str] = {}
            prompt_sha = ""
            for attempt in range(self.max_retries + 1):
                messages = build_messages(
                    list(pending.values()),
                    {it.text_id: contexts[it.text_id] for it in pending.values()},
                    source_locale=source_locale,
                    target_locale=target_locale,
                    previous_errors=errors or None,
                )
                try:
                    answers, prompt_sha = self._call(messages)
                except Exception as exc:  # noqa: BLE001 — зіпсована відповідь → повтор
                    log.warning("translate.llm_error", attempt=attempt, error=str(exc))
                    errors = {code: f"unparseable answer: {exc}" for code in pending}
                    stats.llm_calls += 1
                    continue
                stats.llm_calls += 1
                errors = {}
                for code, item in list(pending.items()):
                    translation = answers.get(code)
                    reason = (
                        "missing in answer"
                        if translation is None
                        else validate_translation(item.source_text, translation)
                    )
                    if reason:
                        errors[code] = reason
                    else:
                        accepted[code] = translation
                        del pending[code]
                if not pending:
                    break

            for code, translation in accepted.items():
                item = by_code[code]
                stats.results.append((code, item.source_text, translation))
                if not dry_run:
                    write_translation(
                        session,
                        text_id=item.text_id,
                        locale_code=target_locale,
                        translated_text=translation,
                        model=self.model_name,
                        prompt_sha256=prompt_sha,
                        tm_chunk_ids=contexts[item.text_id].chunk_ids,
                    )
                stats.written += 1
            for code, item in pending.items():
                if not dry_run:
                    queue_for_review(
                        session,
                        text_id=item.text_id,
                        locale_code=target_locale,
                        source_text=item.source_text,
                        candidate_text=None,
                        reason=errors.get(code, "unknown"),
                    )
                stats.queued_for_review += 1
            session.flush()
            log.info(
                "translate.batch",
                locale=target_locale,
                accepted=len(accepted),
                review=len(pending),
            )
        return stats
