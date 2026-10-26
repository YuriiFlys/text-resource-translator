"""Фабрика моделей вкладень (LangChain Embeddings) + кеш у rag.embedding_cache."""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from trt.config import Settings, get_settings
from trt.models import RagEmbeddingCache

BATCH_SIZE = 64


def normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class HashingEmbeddings(Embeddings):
    """Детермінована локальна модель без залежностей: feature hashing слів і символьних
    триграм у вектор фіксованої розмірності. Для тестів і демо без завантаження моделей;
    схожість поверхнева (лексична), але стабільна та швидка."""

    def __init__(self, size: int) -> None:
        self.size = size

    def _features(self, text: str) -> list[str]:
        lowered = text.lower()
        words = re.findall(r"\w+", lowered)
        feats = [f"w:{w}" for w in words]
        for w in words:
            padded = f"^{w}$"
            feats.extend(f"c:{padded[i : i + 3]}" for i in range(max(1, len(padded) - 2)))
        return feats or ["<empty>"]

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.size
        for feat in self._features(text):
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % self.size
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
        return normalize(vec)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@lru_cache
def get_embeddings(settings: Settings | None = None) -> Embeddings:
    settings = settings or get_settings()
    if settings.embedding_provider in ("hashing", "fake"):
        return HashingEmbeddings(size=settings.embedding_dim)
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model, dimensions=settings.embedding_dim)
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True, "batch_size": BATCH_SIZE},
    )


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CachedEmbedder:
    """Обгортка: спочатку rag.embedding_cache, потім модель батчами, вектори нормалізовані."""

    def __init__(self, session: Session, model: Embeddings, model_name: str, dim: int) -> None:
        self.session = session
        self.model = model
        self.model_name = model_name
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [_key(t) for t in texts]
        cached: dict[str, list[float]] = {}
        if keys:
            rows = self.session.execute(
                select(RagEmbeddingCache.text_sha256, RagEmbeddingCache.embedding).where(
                    RagEmbeddingCache.embedding_model == self.model_name,
                    RagEmbeddingCache.text_sha256.in_(set(keys)),
                )
            ).all()
            cached = {k: list(v) for k, v in rows}

        missing_idx = [i for i, k in enumerate(keys) if k not in cached]
        for start in range(0, len(missing_idx), BATCH_SIZE):
            batch_idx = missing_idx[start : start + BATCH_SIZE]
            vectors = self.model.embed_documents([texts[i] for i in batch_idx])
            for i, vec in zip(batch_idx, vectors, strict=True):
                vec = normalize(vec)
                if len(vec) != self.dim:
                    raise ValueError(
                        f"embedding dim {len(vec)} != EMBEDDING_DIM {self.dim}; "
                        "change EMBEDDING_DIM and re-run `trt migrate`/`trt embed --reindex`"
                    )
                cached[keys[i]] = vec
            if batch_idx:
                stmt = insert(RagEmbeddingCache).values(
                    [
                        {
                            "text_sha256": keys[i],
                            "embedding_model": self.model_name,
                            "embedding": cached[keys[i]],
                        }
                        for i in batch_idx
                    ]
                )
                self.session.execute(stmt.on_conflict_do_nothing())
        return [cached[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        return normalize(self.model.embed_query(text))
