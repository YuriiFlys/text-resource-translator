from __future__ import annotations

from trt.config import Settings


def test_defaults(monkeypatch):
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    s = Settings(_env_file=None)
    assert s.source_locale == "en"
    assert s.embedding_dim == 768
    assert s.embedding_provider == "huggingface"


def test_env_override(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DIM", "1536")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    s = Settings(_env_file=None)
    assert s.embedding_dim == 1536
    assert s.embedding_provider == "openai"
