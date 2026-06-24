"""
Cross-encoder reranker wrapper.
Default: BAAI/bge-reranker-base via fastembed (ONNX runtime).

Why a reranker
--------------
The embedding model (a *bi-encoder*) encodes the query and each chunk
independently, which is fast enough to score a whole collection but only a
coarse relevance signal. A *cross-encoder* reads ``(query, chunk)`` together and
scores their relevance directly — far more accurate at ordering, but too slow to
run over an entire corpus. So we follow the standard retrieve-then-rerank
pattern: over-fetch cheap candidates from the vector store, then reorder the
short list with the cross-encoder.

Stack note
----------
fastembed's ``TextCrossEncoder`` runs on the same onnxruntime that the embedding
layer already uses — no torch, no sentence-transformers, image stays small. The
public interface (``rerank``) mirrors ``embeddings.get_embedding_model`` so the
model can be swapped via the RERANKER_MODEL setting.

Scores
------
Cross-encoder scores are unbounded logits, NOT cosine similarity. Do not compare
them against ``min_relevance_score`` (which is tuned for cosine). They are only
meaningful *relative to each other* for ordering the candidate set.
"""

import logging
from functools import lru_cache
from typing import Protocol

from app.config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Return one relevance score per document, aligned to input order."""
        ...


class BGEReranker:
    """
    BAAI/bge-reranker-base — cross-encoder served via fastembed (ONNX, CPU).
    Lazy-loads the model on first use, like BGEEmbeddings.
    """

    def __init__(self, model_name: str = settings.reranker_model):
        # fastembed >=0.4 exposes the cross-encoder under fastembed.rerank;
        # fall back to the top-level name in case a future layout re-exports it.
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError:  # pragma: no cover - layout fallback
            from fastembed import TextCrossEncoder

        logger.info(
            f"[Reranker] Loading model '{model_name}' via fastembed "
            f"(first run downloads the model)..."
        )
        self.model_name = model_name
        self.model = TextCrossEncoder(model_name=model_name)
        logger.info(f"[Reranker] Model '{model_name}' loaded successfully.")

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        # fastembed's rerank() returns an iterable of float scores, one per doc.
        return [float(s) for s in self.model.rerank(query, documents)]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """Return the reranker singleton (loaded once, on first call)."""
    return BGEReranker()
