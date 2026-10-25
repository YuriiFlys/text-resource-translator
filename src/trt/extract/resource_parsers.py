"""Парсери файлів ресурсів інтерфейсу → список (key, value, comment)."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Entry:
    key: str
    value: str
    comment: str | None = None


def normalize_text(value: str) -> str:
    """NFC + trim. Плейсхолдери і теги не чіпаємо."""
    return unicodedata.normalize("NFC", value).strip()


def _flatten(obj: Any, prefix: str = "") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_flatten(v, f"{prefix}[{i}]"))
    elif isinstance(obj, str):
        out.append((prefix, obj))
    elif obj is not None:
        out.append((prefix, str(obj)))
    return out


def parse_json(data: bytes) -> list[Entry]:
    """i18next / плоский або вкладений JSON."""
    obj = json.loads(data.decode("utf-8"))
    return [Entry(k, normalize_text(v)) for k, v in _flatten(obj) if v.strip()]


_PROP_LINE = re.compile(r"^\s*([^=:\s]+)\s*[=:]\s*(.*)$")
_PROP_ESCAPE = re.compile(r"\\(u[0-9a-fA-F]{4}|.)")
_PROP_SIMPLE = {"n": "\n", "t": "\t", "r": "\r", "f": "\f"}


def _unescape_properties(value: str) -> str:
    """\\uXXXX → символ; \\n \\t → керівні; \\: \\= \\\\ → сам символ."""

    def repl(m: re.Match[str]) -> str:
        esc = m.group(1)
        if esc.startswith("u"):
            return chr(int(esc[1:], 16))
        return _PROP_SIMPLE.get(esc, esc)

    return _PROP_ESCAPE.sub(repl, value)


def parse_properties(data: bytes) -> list[Entry]:
    """Java .properties: key=value, коментарі # або !, продовження рядка через `\\`."""
    entries: list[Entry] = []
    comment: list[str] = []
    pending = ""
    for raw in data.decode("utf-8", errors="replace").splitlines():
        line = pending + (raw.lstrip() if pending else raw)
        if line.rstrip().endswith("\\") and not line.rstrip().endswith("\\\\"):
            pending = line.rstrip()[:-1]
            continue
        pending = ""
        stripped = line.strip()
        if not stripped:
            comment = []
            continue
        if stripped[0] in "#!":
            comment.append(stripped[1:].strip())
            continue
        m = _PROP_LINE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        value = _unescape_properties(value)
        entries.append(Entry(key, normalize_text(value), " ".join(comment) or None))
        comment = []
    return entries


def parse_po(data: bytes) -> list[Entry]:
    import polib

    po = polib.pofile(data.decode("utf-8"))
    entries: list[Entry] = []
    for e in po:
        if e.obsolete or not e.msgid:
            continue
        key = f"{e.msgctxt}|{e.msgid}" if e.msgctxt else e.msgid
        value = e.msgstr or e.msgid  # для .pot беремо джерело
        comment = e.comment or e.tcomment or None
        entries.append(Entry(key, normalize_text(value), comment))
    return entries


def parse_resx(data: bytes) -> list[Entry]:
    from lxml import etree

    root = etree.fromstring(data)
    entries: list[Entry] = []
    for node in root.iterfind("data"):
        name = node.get("name")
        value = node.findtext("value")
        if name and value:
            entries.append(Entry(name, normalize_text(value), node.findtext("comment")))
    return entries


def parse_xliff(data: bytes) -> list[Entry]:
    from lxml import etree

    root = etree.fromstring(data)
    ns = {"x": root.nsmap.get(None, "")} if root.nsmap.get(None) else {}
    pfx = "x:" if ns else ""
    entries: list[Entry] = []
    for unit in root.iterfind(f".//{pfx}trans-unit", ns) or []:
        key = unit.get("id") or unit.get("resname")
        target = unit.findtext(f"{pfx}target", namespaces=ns)
        source = unit.findtext(f"{pfx}source", namespaces=ns)
        value = target or source
        if key and value:
            entries.append(
                Entry(key, normalize_text(value), unit.findtext(f"{pfx}note", namespaces=ns))
            )
    return entries


PARSERS = {
    "json": parse_json,
    "properties": parse_properties,
    "po": parse_po,
    "pot": parse_po,
    "resx": parse_resx,
    "xliff": parse_xliff,
    "xlf": parse_xliff,
}


def parse_resource(fmt: str, data: bytes) -> list[Entry]:
    try:
        parser = PARSERS[fmt.lower().lstrip(".")]
    except KeyError as exc:
        raise ValueError(f"unsupported resource format: {fmt}") from exc
    return parser(data)
