"""Побудова промпту для LLM: правила + translation memory + глосарій + пакет рядків."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict


@dataclass
class WorkItem:
    text_id: int
    text_code: str | None
    owner_kind: str | None
    source_text: str


@dataclass
class TMExample:
    chunk_id: int
    source_text: str
    target_text: str
    text_code: str | None
    score: float


@dataclass
class TranslationContext:
    tm_examples: list[TMExample] = field(default_factory=list)
    glossary_snippets: list[tuple[int, str]] = field(default_factory=list)  # (chunk_id, text)

    @property
    def chunk_ids(self) -> list[int]:
        return [e.chunk_id for e in self.tm_examples] + [cid for cid, _ in self.glossary_snippets]


class TranslationBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: dict[str, str]


def batch_schema(count: int) -> dict:
    """Оllama grammar: рівно один переклад для кожного ID поточного запиту."""
    ids = [str(i) for i in range(count)]
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "object",
                "properties": {key: {"type": "string"} for key in ids},
                "required": ids,
                "additionalProperties": False,
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


SYSTEM_PROMPT = """You are a professional software localization specialist.
Translate user-interface strings from {source_locale} into {target_locale}.

Rules:
1. Preserve every placeholder and markup token exactly as in the source:
   {{0}}, {{name}}, {{{{var}}}}, %s, %1$s, ${{var}}, HTML tags, \\n.
2. Keep the meaning, register and tone of the source. UI strings must be concise;
   keep length within about ±30% of the source unless the target language requires otherwise.
3. Use the terminology from the glossary and be consistent with the translation memory
   examples. Prefer the exact wording of a memory example when the source is identical.
4. Respect the element type (owner_kind): buttons are short imperative labels,
   headings are nouns, errors are full sentences.
5. Do not translate product names, codes or brand names.
6. Return ONLY a JSON object mapping each input id to its translation:
   {{"items": {{"0": "translation for id 0", "1": "translation for id 1"}}}}
   Include every input id exactly once. No commentary, no code fences.
"""


def build_messages(
    items: list[WorkItem],
    contexts: dict[int, TranslationContext],
    *,
    source_locale: str,
    target_locale: str,
    previous_errors: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    tm: dict[str, dict] = {}
    glossary: dict[int, str] = {}
    for ctx in contexts.values():
        for ex in ctx.tm_examples:
            tm.setdefault(ex.source_text, {"source": ex.source_text, "target": ex.target_text})
        for cid, snippet in ctx.glossary_snippets:
            glossary.setdefault(cid, snippet)

    payload = {
        "source_locale": source_locale,
        "target_locale": target_locale,
        "translation_memory": list(tm.values())[:40],
        "glossary_and_style_guide": list(glossary.values())[:8],
        "strings_to_translate": [
            {
                "id": str(i),
                "text_code": it.text_code or f"text_id:{it.text_id}",
                "owner_kind": it.owner_kind,
                "source": it.source_text,
            }
            for i, it in enumerate(items)
        ],
    }
    if previous_errors:
        payload["previous_attempt_errors"] = previous_errors
        payload["instruction"] = (
            "Your previous answer was rejected for the listed text_codes. Fix those errors."
        )

    system = SYSTEM_PROMPT.format(source_locale=source_locale, target_locale=target_locale)
    return [
        ("system", system),
        ("human", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
    ]
