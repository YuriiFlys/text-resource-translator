from __future__ import annotations

from pathlib import Path

from trt.extract.document_text import extract_document
from trt.extract.resource_parsers import parse_resource

ROOT = Path(__file__).resolve().parents[1]


def test_json_flatten():
    entries = parse_resource("json", (ROOT / "samples" / "web-ui.en.json").read_bytes())
    by_key = {e.key: e.value for e in entries}
    assert by_key["nav.signOut"] == "Sign out"
    assert by_key["projects.members"] == "{{count}} members"
    assert len(entries) == 10


def test_properties_with_comments_and_placeholders():
    entries = parse_resource("properties", (ROOT / "samples" / "mobile_en.properties").read_bytes())
    by_key = {e.key: e for e in entries}
    assert by_key["sync.now"].value == "Sync now"
    assert by_key["sync.now"].comment == "Shown on the main button"
    assert by_key["storage.used"].value == "%1$s of %2$s used"
    assert by_key["sync.last"].comment is None  # коментар належить лише наступному ключу


def test_properties_unicode_escapes():
    data = b"greeting=\\u041f\\u0440\\u0438\\u0432\\u0456\\u0442\nmulti=one \\\n  two\n"
    by_key = {e.key: e.value for e in parse_resource("properties", data)}
    assert by_key["greeting"] == "Привіт"
    assert by_key["multi"] == "one two"


def test_po():
    po = """msgid ""
msgstr ""
"Content-Type: text/plain; charset=UTF-8\\n"

#. button label
msgid "Save"
msgstr "Зберегти"

msgctxt "menu"
msgid "Open"
msgstr ""
""".encode()
    entries = parse_resource("po", po)
    assert entries[0].key == "Save" and entries[0].value == "Зберегти"
    assert entries[0].comment == "button label"
    assert entries[1].key == "menu|Open" and entries[1].value == "Open"


def test_resx_and_xliff():
    resx = b"""<?xml version="1.0"?><root>
      <data name="Save" xml:space="preserve"><value>Save</value><comment>btn</comment></data>
      <data name="Empty"><value></value></data></root>"""
    assert [(e.key, e.value, e.comment) for e in parse_resource("resx", resx)] == [
        ("Save", "Save", "btn")
    ]
    xliff = """<?xml version="1.0"?>
    <xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" version="1.2"><file><body>
      <trans-unit id="save"><source>Save</source><target>Зберегти</target></trans-unit>
      <trans-unit id="cancel"><source>Cancel</source></trans-unit>
    </body></file></xliff>""".encode()
    by_key = {e.key: e.value for e in parse_resource("xliff", xliff)}
    assert by_key == {"save": "Зберегти", "cancel": "Cancel"}


def test_pdf_and_html_extraction():
    pdf = extract_document("pdf", (ROOT / "samples" / "style-guide.pdf").read_bytes())
    assert "Localization Style Guide" in pdf.text
    assert pdf.pages and pdf.pages[0]["page"] == 1

    raw = b"<html><style>x{}</style><body><h1>Hi</h1><p>A &amp; B</p></body></html>"
    html = extract_document("html", raw)
    assert "Hi" in html.text and "A &amp; B" in html.text and "x{}" not in html.text
