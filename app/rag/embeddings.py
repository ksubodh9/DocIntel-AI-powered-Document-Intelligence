"""
Embedding model wrapper.
Default: BAAI/bge-small-en-v1.5 via fastembed (ONNX runtime).

fastembed replaces torch + sentence-transformers: it runs the same model on
onnxruntime (which ChromaDB already bundles), cutting the image size and RAM
footprint by well over a gigabyte. That's what lets the backend fit on small
free tiers. The public interface (embed_documents / embed_query) is unchanged,
so the chunker, vectorstore, and document service are untouched.

BGE models prepend a query instruction for retrieval queries:
  "Represent this sentence for searching relevant passages: <query>"
This is handled automatically in embed_query().
"""

from functools import lru_cache
from typing import Protocol

from app.config.settings import get_settings

settings = get_settings()


class EmbeddingModel(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


# ── HuggingFace / fastembed (ONNX) Embeddings ────────────────────────────────

class BGEEmbeddings:
    """
    BAAI/bge-small-en-v1.5 — fast, lightweight, 384-dim, served via fastembed.
    Uses the recommended query prefix for retrieval tasks. Vectors are
    L2-normalized so cosine similarity in ChromaDB behaves as before.
    """

    BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = settings.embedding_model, device: str = settings.embedding_device):
        import logging
        from fastembed import TextEmbedding
        _log = logging.getLogger(__name__)
        _log.info(f"[Embeddings] Loading model '{model_name}' via fastembed (first run downloads ~130 MB)...")
        self.model_name = model_name
        # device is accepted for interface compatibility; fastembed runs on CPU
        # via onnxruntime. Install fastembed-gpu to use a GPU.
        self.model = TextEmbedding(model_name=model_name)
        self._is_bge = "bge" in model_name.lower()
        _log.info(f"[Embeddings] Model '{model_name}' loaded successfully.")

    @staticmethod
    def _normalize(vec) -> list[float]:
        import numpy as np
        v = np.asarray(vec, dtype="float32")
        norm = float(np.linalg.norm(v))
        return (v / norm).tolist() if norm else v.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # fastembed.embed() returns a generator of numpy arrays.
        return [self._normalize(e) for e in self.model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        if self._is_bge:
            text = self.BGE_QUERY_INSTRUCTION + text
        emb = next(iter(self.model.embed([text])))
        return self._normalize(emb)


# ── OpenAI Embeddings ─────────────────────────────────────────────────────────

class OpenAIEmbeddings:
    """OpenAI text-embedding-3-small — 1536-dim, requires API key."""

    def __init__(self, api_key: str = settings.openai_api_key, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [d.embedding for d in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding


# ── Factory ───────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """
    Return the embedding model singleton.
    Determines type based on EMBEDDING_MODEL setting:
      - Starts with "BAAI/" or "sentence-transformers/" → BGEEmbeddings
      - "text-embedding-*" → OpenAIEmbeddings
    """
    model_name = settings.embedding_model.lower()
    if model_name.startswith("text-embedding"):
        return OpenAIEmbeddings()
    return BGEEmbeddings()
