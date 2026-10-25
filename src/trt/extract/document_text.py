"""Витягування тексту з довідкових документів (PDF, DOCX, MD, HTML, TXT)."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

from trt.extract.resource_parsers import normalize_text


@dataclass
class ExtractedDocument:
    text: str
    pages: list[dict] | None = None  # [{"page": 1, "text": "..."}]


def extract_pdf(data: bytes) -> ExtractedDocument:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if text:
            pages.append({"page": i, "text": text})
    return ExtractedDocument(text="\n\n".join(p["text"] for p in pages), pages=pages)


def extract_docx(data: bytes) -> ExtractedDocument:
    import docx2txt

    return ExtractedDocument(text=normalize_text(docx2txt.process(io.BytesIO(data))))


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t]+")


def extract_html(data: bytes) -> ExtractedDocument:
    html = data.decode("utf-8", errors="replace")
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</h\d>|</li>", "\n", html)
    text = _TAG.sub(" ", html)
    text = "\n".join(_WS.sub(" ", line).strip() for line in text.splitlines())
    return ExtractedDocument(text=normalize_text(re.sub(r"\n{3,}", "\n\n", text)))


def extract_plain(data: bytes) -> ExtractedDocument:
    return ExtractedDocument(text=normalize_text(data.decode("utf-8", errors="replace")))


EXTRACTORS = {
    "pdf": extract_pdf,
    "docx": extract_docx,
    "html": extract_html,
    "htm": extract_html,
    "md": extract_plain,
    "markdown": extract_plain,
    "txt": extract_plain,
}


def extract_document(fmt: str, data: bytes) -> ExtractedDocument:
    try:
        extractor = EXTRACTORS[fmt.lower().lstrip(".")]
    except KeyError as exc:
        raise ValueError(f"unsupported document format: {fmt}") from exc
    return extractor(data)
