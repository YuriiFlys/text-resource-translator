"""Перевірка перекладу: плейсхолдери й теги мають збігатися з джерелом."""

from __future__ import annotations

import re
from collections import Counter

PLACEHOLDER_RE = re.compile(
    r"\{\{\s*[\w.]+\s*\}\}"  # {{name}}
    r"|\{[\w.]*\}"  # {0}, {name}, {}
    r"|%\d*\$?[sdif]"  # %s, %d, %1$s
    r"|\$\{[\w.]+\}"  # ${var}
    r"|</?[a-zA-Z][\w-]*(?:\s[^<>]*)?/?>"  # <b>, </a>, <br/>
    r"|\\n|\\t"  # літеральні \n \t
)


def placeholders(text: str) -> Counter[str]:
    return Counter(PLACEHOLDER_RE.findall(text))


def is_literal(text: str) -> bool:
    """Email, URL і рядки лише з плейсхолдерів не потребують перекладу."""
    value = text.strip()
    return bool(
        re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value)
        or re.fullmatch(r"https?://\S+", value)
        or (value and not PLACEHOLDER_RE.sub("", value).strip())
    )


def validate_translation(source: str, translation: str) -> str | None:
    """None, якщо переклад прийнятний; інакше — причина відхилення."""
    if not translation or not translation.strip():
        return "empty translation"
    src_ph, tr_ph = placeholders(source), placeholders(translation)
    if src_ph != tr_ph:
        missing = list((src_ph - tr_ph).elements())
        extra = list((tr_ph - src_ph).elements())
        return f"placeholder mismatch: missing={missing} extra={extra}"
    if (
        translation.strip() == source.strip()
        and not is_literal(source)
        and re.search(r"[A-Za-zА-Яа-яІіЇїЄєҐґ]{3,}", source)
    ):
        return "translation identical to source"
    if len(translation) > max(40, len(source) * 3):
        return "translation suspiciously long"
    return None
