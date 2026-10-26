"""LLM-переклад з контекстом із векторного пошуку (translation memory + глосарій)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from time import perf_counter

from langchain_core.language_models import BaseChatModel
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from trt.config import get_settings
from trt.logging import get_logger
from trt.models import I18nText, I18nTranslation, RagChunk
from trt.translate.prompt import (
    TMExample,
    TranslationBatch,
    TranslationContext,
    WorkItem,
    batch_schema,
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
    accepted: int = 0
    written: int = 0
    queued_for_review: int = 0
    llm_calls: int = 0
    elapsed_seconds: float = 0
    review_reasons: dict[str, str] = field(default_factory=dict)
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
    query_vector: list[float] | None = None,
) -> TranslationContext:
    """kNN серед UI-рядків вихідної локалі → їхні наявні переклади = translation memory."""
    query_vec = query_vector
    if query_vec is None:
        query_vec = make_embedder(session).embed_query(context_query(item))

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


def context_query(item: WorkItem) -> str:
    prefix = " | ".join(p for p in (item.text_code, item.owner_kind) if p)
    return f"{prefix}: {item.source_text}" if prefix else item.source_text


def build_contexts(
    session: Session,
    items: list[WorkItem],
    *,
    source_locale: str,
    target_locale: str,
) -> dict[int, TranslationContext]:
    """Без індексу не завантажувати модель; інакше векторизувати пакет із кешем."""
    has_context = session.scalar(
        select(RagChunk.chunk_id)
        .where(
            (RagChunk.kind == "reference_doc")
            | ((RagChunk.kind == "ui_string") & (RagChunk.locale_code == source_locale))
        )
        .limit(1)
    )
    if has_context is None:
        return {item.text_id: TranslationContext() for item in items}

    vectors = make_embedder(session).embed_documents([context_query(item) for item in items])
    return {
        item.text_id: build_context(
            session,
            item,
            source_locale=source_locale,
            target_locale=target_locale,
            query_vector=vector,
        )
        for item, vector in zip(items, vectors, strict=True)
    }


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_batch(raw: str) -> TranslationBatch:
    cleaned = _FENCE.sub("", raw.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("LLM answer contains no JSON object")
    return TranslationBatch.model_validate_json(cleaned[start : end + 1])


def get_chat_model(model: str | None = None) -> BaseChatModel:
    settings = get_settings()
    model_name = model or settings.llm_model
    provider = settings.llm_provider
    if provider == "ollama" or (provider == "auto" and model_name.startswith("ollama:")):
        from langchain_ollama import ChatOllama

        ollama_model = model_name.removeprefix("ollama:")
        return ChatOllama(
            model=ollama_model,
            base_url=settings.ollama_base_url,
            reasoning=False,
            temperature=0,
            format=TranslationBatch.model_json_schema(),
            keep_alive="10m",
            client_kwargs={"timeout": settings.ollama_timeout_seconds},
        )

    from langchain.chat_models import init_chat_model

    if provider != "auto":
        return init_chat_model(model_name, model_provider=provider)
    # Провайдер виводиться з назви моделі (claude-* → anthropic, gpt-* → openai, ...).
    return init_chat_model(model_name)


class Translator:
    def __init__(
        self,
        llm: BaseChatModel | None = None,
        *,
        batch_size: int | None = None,
        max_retries: int | None = None,
    ):
        settings = get_settings()
        self._llm = llm
        self.batch_size = settings.translation_batch_size if batch_size is None else batch_size
        self.max_retries = settings.translation_max_retries if max_retries is None else max_retries
        if self.batch_size < 1 or self.max_retries < 0:
            raise ValueError("batch_size must be positive and max_retries nonnegative")
        self.model_name = settings.llm_model
        self._ollama = llm is None and (
            settings.llm_provider == "ollama"
            or (settings.llm_provider == "auto" and self.model_name.startswith("ollama:"))
        )

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = get_chat_model()
        return self._llm

    def _call(self, messages: list[tuple[str, str]]) -> tuple[dict[str, str], str]:
        options = {}
        if self._ollama:
            count = len(json.loads(messages[-1][1])["strings_to_translate"])
            options["format"] = batch_schema(count)
        started = perf_counter()
        response = self.llm.invoke(messages, **options)
        log.info(
            "translate.llm_done",
            seconds=round(perf_counter() - started, 2),
            input_tokens=response.response_metadata.get("prompt_eval_count"),
            output_tokens=response.response_metadata.get("eval_count"),
        )
        if response.response_metadata.get("done_reason") == "length":
            raise ValueError("LLM output was truncated; reduce TRANSLATION_BATCH_SIZE")
        content = (
            response.content if isinstance(response.content, str) else json.dumps(response.content)
        )
        batch = parse_batch(content)
        prompt_sha = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()
        return batch.items, prompt_sha

    def translate_locale(
        self, session: Session, target_locale: str, *, limit: int = 100, dry_run: bool = False
    ) -> TranslateStats:
        settings = get_settings()
        started = perf_counter()
        source_locale = settings.source_locale
        stats = TranslateStats(locale=target_locale)
        items = select_work(session, source_locale, target_locale, limit)
        stats.requested = len(items)
        if not items:
            return stats

        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            by_code = {it.text_code or f"text_id:{it.text_id}": it for it in batch}
            log.info(
                "translate.context_start", locale=target_locale, offset=start, count=len(batch)
            )
            contexts = build_contexts(
                session, batch, source_locale=source_locale, target_locale=target_locale
            )
            pending = dict(by_code)
            errors: dict[str, str] = {}
            accepted: dict[str, str] = {}
            accepted_prompts: dict[str, str] = {}
            candidates: dict[str, str] = {}
            prompt_sha = ""
            for attempt in range(self.max_retries + 1):
                messages = build_messages(
                    list(pending.values()),
                    {it.text_id: contexts[it.text_id] for it in pending.values()},
                    source_locale=source_locale,
                    target_locale=target_locale,
                    previous_errors=errors or None,
                )
                log.info(
                    "translate.llm_start", locale=target_locale, count=len(pending), attempt=attempt
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
                for index, (code, item) in enumerate(list(pending.items())):
                    translation = answers.get(str(index))
                    if translation is not None:
                        candidates[code] = translation
                    reason = (
                        "missing in answer"
                        if translation is None
                        else validate_translation(item.source_text, translation)
                    )
                    if reason:
                        errors[code] = reason
                    else:
                        accepted[code] = translation
                        accepted_prompts[code] = prompt_sha
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
                        prompt_sha256=accepted_prompts[code],
                        tm_chunk_ids=contexts[item.text_id].chunk_ids,
                    )
                    stats.written += 1
                stats.accepted += 1
            for code, item in pending.items():
                if not dry_run:
                    queue_for_review(
                        session,
                        text_id=item.text_id,
                        locale_code=target_locale,
                        source_text=item.source_text,
                        candidate_text=candidates.get(code),
                        reason=errors.get(code, "unknown"),
                    )
                stats.queued_for_review += 1
                stats.review_reasons[code] = errors.get(code, "unknown")
            session.flush()
            log.info(
                "translate.batch",
                locale=target_locale,
                accepted=len(accepted),
                review=len(pending),
                completed=start + len(batch),
                total=len(items),
            )
        stats.elapsed_seconds = round(perf_counter() - started, 2)
        return stats
