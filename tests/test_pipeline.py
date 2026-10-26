"""Інтеграційний сценарій: fetch → ingest → embed → search → translate (fake LLM) → writeback."""

from __future__ import annotations

from sqlalchemy import text

from tests.conftest import requires_pg
from tests.fakes import DictionaryFakeChat

UK = {
    "Delete": "Видалити",
    "Changes are saved automatically every {0} seconds.": (
        "Зміни зберігаються автоматично кожні {0} секунд."
    ),
    "Are you sure you want to delete your profile? This action cannot be undone.": (
        "Ви впевнені, що хочете видалити профіль? Цю дію не можна скасувати."
    ),
    "Invalid username or password.": "Неправильне ім'я користувача або пароль.",
    "Uploading {{fileName}}: {{percent}}%": "Завантаження {{fileName}}: {{percent}}%",
}


@requires_pg
def test_full_pipeline(pg_env):
    from trt.db import session_scope
    from trt.pipeline import run_all, step_embed, step_fetch, step_ingest
    from trt.vector_store import search_text

    # --- fetch: знімок i18n + файли ресурсів + документи на диску ---
    fetched = {r.source: r for r in step_fetch("all")}
    assert len(fetched["postgres"].artifacts) == 1
    assert len(fetched["resource_file"].artifacts) == 3
    assert len(fetched["document"].artifacts) == 2
    assert fetched["postgres"].artifacts[0].path.endswith(".jsonl")

    # --- ingest: повні тексти в rag.document ---
    ingest = step_ingest()
    assert ingest.errors == []
    assert ingest.upserted >= 15 + 10 + 7 + 3 + 2

    # --- embed: фрагменти + вектори ---
    emb = step_embed()
    assert emb.documents == ingest.upserted and emb.chunks >= emb.documents

    # повторний прогін нічого не змінює
    assert all(not r.artifacts for r in step_fetch("all"))
    assert step_ingest().upserted == 0
    assert step_embed().chunks == 0

    # --- search: kNN повертає текст, chunk_id, document_id ---
    with session_scope() as s:
        hits = search_text(s, "Save", 3, locale="en", kind="ui_string")
        assert hits and hits[0].text == "Save" and hits[0].chunk_id and hits[0].document_id
        assert hits[0].score > hits[-1].score
        glossary = search_text(s, "глосарій project проєкт", 1, kind="reference_doc")
        assert glossary and glossary[0].doc_key == "doc:glossary"

    # --- translate через фейкову LLM; один рядок спочатку ламає плейсхолдер → retry ---
    llm = DictionaryFakeChat(dictionary=UK, break_once={"settings.profile.save_hint"}, fence=True)
    report = run_all(locales=["uk"], llm=llm)
    assert not report.skipped_locked
    uk = report.translated[0]
    assert uk.requested == 5 and uk.written == 5 and uk.queued_for_review == 0
    assert llm.calls == 2  # основний виклик + повтор для зіпсованого рядка

    with session_scope() as s:
        rows = s.execute(
            text(
                "SELECT t.text_code, tr.translated_text FROM i18n_translation tr "
                "JOIN i18n_text t USING (text_id) WHERE tr.locale_code = 'uk' ORDER BY t.text_code"
            )
        ).all()
        assert (
            dict(rows)["settings.profile.save_hint"]
            == UK["Changes are saved automatically every {0} seconds."]
        )
        assert len(rows) == 10  # 5 seed + 5 нових
        audit = s.execute(text("SELECT count(*) FROM rag.translation_audit")).scalar_one()
        assert audit == 5
        # нові переклади вже стали частиною translation memory (документи + вектори)
        tm_docs = s.execute(
            text("SELECT count(*) FROM rag.document WHERE locale_code='uk' AND source='postgres'")
        ).scalar_one()
        assert tm_docs == 10

    # повторний run-all: роботи немає
    report2 = run_all(locales=["uk"], llm=llm)
    assert report2.translated[0].requested == 0 and llm.calls == 2


@requires_pg
def test_translate_queue_for_review_when_llm_keeps_failing(pg_env):
    from trt.db import session_scope
    from trt.translate.translator import Translator

    class AlwaysBroken(DictionaryFakeChat):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.calls += 1
            from langchain_core.messages import AIMessage
            from langchain_core.outputs import ChatGeneration, ChatResult

            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="not json"))])

    with session_scope() as s:
        stats = Translator(AlwaysBroken(dictionary={}), max_retries=1).translate_locale(
            s, "pl", limit=3
        )
        assert stats.requested == 3 and stats.written == 0 and stats.queued_for_review == 3
        queued = s.execute(
            text("SELECT count(*) FROM rag.review_queue WHERE locale_code='pl'")
        ).scalar_one()
        assert queued == 3
        assert (
            s.execute(
                text("SELECT count(*) FROM i18n_translation WHERE locale_code='pl'")
            ).scalar_one()
            == 0
        )
