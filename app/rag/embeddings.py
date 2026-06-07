"""
Embedding model wrapper.
Default: BAAI/bge-small-en-v1.5 via sentence-transformers.
Modular: swap to OpenAI embeddings by changing EMBEDDING_PROVIDER in .env.

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


# ── HuggingFace / Sentence-Transformers Embeddings ───────────────────────────

class BGEEmbeddings:
    """
    BAAI/bge-small-en-v1.5 — fast, lightweight, 384-dim.
    Uses the recommended query prefix for retrieval tasks.
    """

    BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = settings.embedding_model, device: str = settings.embedding_device):
        import logging
        from sentence_transformers import SentenceTransformer
        _log = logging.getLogger(__name__)
        _log.info(f"[Embeddings] Loading model '{model_name}' on {device} (first run downloads ~130 MB)...")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self._is_bge = "bge" in model_name.lower()
        _log.info(f"[Embeddings] Model '{model_name}' loaded successfully.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        if self._is_bge:
            text = self.BGE_QUERY_INSTRUCTION + text
        embedding = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return embedding[0].tolist()


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
