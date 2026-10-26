from __future__ import annotations

import json

from trt.chunking import embed_text_for, split_document
from trt.models import RagDocument
from trt.translate.prompt import WorkItem, build_messages
from trt.translate.translator import parse_batch
from trt.translate.validation import placeholders, validate_translation


def _doc(**kw) -> RagDocument:
    base = dict(document_id=1, doc_key="k", kind="ui_string", locale_code="en", text_id=1)
    return RagDocument(**{**base, **kw})


def test_ui_string_is_single_chunk_with_context():
    doc = _doc(content="Save", text_code="common.save", owner_kind="button")
    chunks = split_document(doc)
    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_index"] == 0
    assert embed_text_for(doc, "Save") == "common.save | button: Save"


def test_long_reference_doc_is_split_with_overlap():
    para = "Sentence number {} of the style guide explains a rule in detail. "
    content = "\n\n".join("".join(para.format(i) for i in range(k, k + 6)) for k in range(0, 60, 6))
    doc = _doc(kind="reference_doc", content=content, raw_path="x.txt")
    chunks = split_document(doc)
    assert len(chunks) > 3
    assert all(len(c.page_content) <= 800 for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_markdown_split_by_headers():
    content = "# Title\n\nintro\n\n## A\n\nterm a\n\n## B\n\nterm b\n"
    chunks = split_document(_doc(kind="reference_doc", content=content, raw_path="g.md"))
    assert len(chunks) == 3
    assert chunks[1].page_content.startswith("## A")


def test_placeholders_and_validation():
    src = "Uploading {{fileName}}: {{percent}}% of %1$s <b>{0}</b>"
    assert sorted(placeholders(src).elements()) == sorted(
        ["{{fileName}}", "{{percent}}", "%1$s", "<b>", "</b>", "{0}"]
    )
    assert validate_translation(src, src.replace("Uploading", "Завантаження")) is None
    assert "missing" in validate_translation(src, "Завантаження {{fileName}}")
    assert validate_translation("Save", "") == "empty translation"
    assert validate_translation("Save", "Save") == "translation identical to source"
    assert validate_translation("OK", "OK") is None  # короткі коди/абревіатури можуть збігатися


def test_build_messages_and_parse_batch():
    items = [WorkItem(1, "common.save", "button", "Save")]
    msgs = build_messages(items, {}, source_locale="en", target_locale="uk")
    assert msgs[0][0] == "system" and "uk" in msgs[0][1]
    assert json.loads(msgs[1][1])["strings_to_translate"][0]["id"] == "0"
    batch = parse_batch('```json\n{"items": {"0": "Зберегти"}}\n```')
    assert batch.items["0"] == "Зберегти"
