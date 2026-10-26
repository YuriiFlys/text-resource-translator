from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from tests.fakes import DictionaryFakeChat
from trt.translate import translator as module
from trt.translate.prompt import WorkItem, batch_schema, build_messages
from trt.translate.validation import validate_translation


def test_empty_index_does_not_load_embedding_model(monkeypatch):
    session = Mock()
    session.scalar.return_value = None
    embedder = Mock(side_effect=AssertionError("empty index must not load an embedding model"))
    monkeypatch.setattr(module, "make_embedder", embedder)
    items = [WorkItem(1, "save", "button", "Save")]
    contexts = module.build_contexts(session, items, source_locale="en", target_locale="uk")
    assert contexts[1].tm_examples == []
    embedder.assert_not_called()


def test_context_queries_are_embedded_in_one_cached_batch(monkeypatch):
    session = Mock()
    session.scalar.return_value = 1
    embedder = Mock()
    embedder.embed_documents.return_value = [[1.0], [2.0]]
    monkeypatch.setattr(module, "make_embedder", lambda _: embedder)
    search = Mock(return_value=[])
    monkeypatch.setattr(module, "search_similar", search)
    items = [WorkItem(1, "save", "button", "Save"), WorkItem(2, None, None, "Cancel")]
    contexts = module.build_contexts(session, items, source_locale="en", target_locale="uk")
    embedder.embed_documents.assert_called_once_with(["save | button: Save", "Cancel"])
    assert set(contexts) == {1, 2}
    assert [call.args[1] for call in search.call_args_list] == [[1.0], [1.0], [2.0], [2.0]]


@pytest.mark.parametrize("source", ["name@example.com", "https://example.com/help", "{user}"])
def test_literals_can_remain_unchanged(source):
    assert validate_translation(source, source) is None
    assert validate_translation("Sign in", "Sign in") == "translation identical to source"


def prepare_translation(monkeypatch, items):
    monkeypatch.setattr(module, "select_work", lambda *_: items)
    monkeypatch.setattr(
        module,
        "build_contexts",
        lambda _session, batch, **_: {item.text_id: module.TranslationContext() for item in batch},
    )
    write = Mock()
    queue = Mock()
    monkeypatch.setattr(module, "write_translation", write)
    monkeypatch.setattr(module, "queue_for_review", queue)
    return write, queue


def test_retry_renumbers_pending_ids_and_keeps_original_prompt_hash(monkeypatch):
    items = [
        WorkItem(1, "save", "button", "Save"),
        WorkItem(2, "welcome", "hint", "Welcome {name}"),
    ]
    write, queue = prepare_translation(monkeypatch, items)
    llm = DictionaryFakeChat(
        dictionary={"Save": "Зберегти", "Welcome {name}": "Вітаємо {name}"},
        break_once={"welcome"},
    )
    stats = module.Translator(llm).translate_locale(Mock(), "uk")
    assert stats.accepted == stats.written == 2 and stats.llm_calls == 2
    assert stats.queued_for_review == 0
    writes = {call.kwargs["text_id"]: call.kwargs for call in write.call_args_list}
    assert writes[1]["translated_text"] == "Зберегти"
    assert writes[2]["translated_text"] == "Вітаємо {name}"
    assert writes[1]["prompt_sha256"] != writes[2]["prompt_sha256"]
    queue.assert_not_called()


def test_dry_run_reports_accepted_separately_from_written(monkeypatch):
    write, queue = prepare_translation(monkeypatch, [WorkItem(1, "save", "button", "Save")])
    stats = module.Translator(DictionaryFakeChat(dictionary={"Save": "Зберегти"})).translate_locale(
        Mock(), "uk", dry_run=True
    )
    assert stats.accepted == 1 and stats.written == 0
    assert stats.results == [("save", "Save", "Зберегти")]
    write.assert_not_called()
    queue.assert_not_called()


def test_review_preserves_failed_candidate_and_reason(monkeypatch):
    write, queue = prepare_translation(
        monkeypatch, [WorkItem(1, "welcome", "hint", "Welcome {name}")]
    )
    stats = module.Translator(
        DictionaryFakeChat(dictionary={"Welcome {name}": "Вітаємо"}), max_retries=0
    ).translate_locale(Mock(), "uk")
    assert stats.accepted == 0 and stats.queued_for_review == 1
    assert "placeholder mismatch" in stats.review_reasons["welcome"]
    assert queue.call_args.kwargs["candidate_text"] == "Вітаємо"
    write.assert_not_called()


def test_native_ollama_uses_schema_for_exact_pending_ids(monkeypatch):
    llm = Mock()
    llm.invoke.return_value = AIMessage(content='{"items":{"0":"Зберегти"}}')
    monkeypatch.setattr(module, "get_chat_model", lambda: llm)
    monkeypatch.setattr(
        module,
        "get_settings",
        lambda: SimpleNamespace(
            llm_model="qwen3.5:9b",
            llm_provider="ollama",
            translation_batch_size=20,
            translation_max_retries=1,
        ),
    )
    messages = build_messages(
        [WorkItem(1, "save", "button", "Save")], {}, source_locale="en", target_locale="uk"
    )
    answers, _ = module.Translator()._call(messages)
    assert answers == {"0": "Зберегти"}
    schema = llm.invoke.call_args.kwargs["format"]
    assert schema == batch_schema(1)
    assert schema["properties"]["items"]["required"] == ["0"]
    assert schema["properties"]["items"]["additionalProperties"] is False
    assert json.loads(messages[-1][1])["strings_to_translate"][0]["text_code"] == "save"


def test_truncated_response_is_reported_before_json_parsing():
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content='{"items":{"0":', response_metadata={"done_reason": "length"}
    )
    with pytest.raises(ValueError, match="truncated"):
        module.Translator(llm)._call([("human", "test")])
