"""Розбиття документів на фрагменти засобами LangChain."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from trt.models import RagDocument

UI_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
DOC_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""],
)
MD_HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=False,
)


def embed_text_for(doc: RagDocument, chunk_text: str) -> str:
    """Текст, який подається у модель вкладень: для UI-рядків збагачений контекстом."""
    if doc.kind == "ui_string":
        prefix = " | ".join(p for p in (doc.text_code, doc.owner_kind) if p)
        return f"{prefix}: {chunk_text}" if prefix else chunk_text
    return chunk_text


def split_document(doc: RagDocument) -> list[Document]:
    """Повертає LangChain Document'и з metadata.chunk_index та посиланням на документ."""
    base_meta = {
        "document_id": doc.document_id,
        "doc_key": doc.doc_key,
        "kind": doc.kind,
        "locale_code": doc.locale_code,
        "text_id": doc.text_id,
        "owner_kind": doc.owner_kind,
    }
    if doc.kind == "ui_string":
        pieces = UI_SPLITTER.split_text(doc.content)
    elif doc.raw_path and doc.raw_path.lower().endswith((".md", ".markdown")):
        sections = MD_HEADER_SPLITTER.split_text(doc.content)
        pieces = [c.page_content for c in DOC_SPLITTER.split_documents(sections)]
    else:
        pieces = DOC_SPLITTER.split_text(doc.content)

    return [
        Document(page_content=piece, metadata={**base_meta, "chunk_index": i})
        for i, piece in enumerate(p for p in pieces if p.strip())
    ]
