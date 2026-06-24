"""
Unit tests for the retrieve-then-rerank stage.

These tests mock both the embedding model and the cross-encoder so they run
fast and offline — we verify the *wiring* (over-fetch, reorder, truncate,
fallback), not the quality of any real model.
"""

from unittest.mock import patch, MagicMock

import app.rag.vectorstore as vs


def _fake_chunks(n: int) -> list[dict]:
    """Candidate chunks as retrieve_chunks builds them, in cosine order."""
    return [
        {
            "text": f"chunk {i}",
            "page_number": i,
            "chunk_id": f"c{i}",
            "distance": round(0.1 * i, 4),
            "relevance_score": round(1 - 0.1 * i, 4),
        }
        for i in range(n)
    ]


class TestApplyRerank:
    def test_reorders_by_rerank_score_and_truncates(self):
        chunks = _fake_chunks(4)  # cosine order: c0, c1, c2, c3
        # Reranker thinks the LAST candidate is the most relevant.
        fake = MagicMock()
        fake.rerank.return_value = [0.1, 0.2, 0.3, 0.9]
        with patch("app.rag.reranker.get_reranker", return_value=fake):
            out = vs._apply_rerank("q", chunks, top_k=2)

        assert len(out) == 2                      # truncated to top_k
        assert out[0]["chunk_id"] == "c3"         # highest rerank_score first
        assert out[1]["chunk_id"] == "c2"
        assert out[0]["rerank_score"] == 0.9
        # Original cosine score is preserved so the downstream threshold still works.
        assert "relevance_score" in out[0]

    def test_falls_back_to_cosine_order_on_reranker_error(self):
        chunks = _fake_chunks(3)
        fake = MagicMock()
        fake.rerank.side_effect = RuntimeError("model unavailable")
        with patch("app.rag.reranker.get_reranker", return_value=fake):
            out = vs._apply_rerank("q", chunks, top_k=3)

        # Falls back to the incoming cosine order, never raises.
        assert [c["chunk_id"] for c in out] == ["c0", "c1", "c2"]

    def test_empty_input_returns_empty(self):
        assert vs._apply_rerank("q", [], top_k=5) == []


class TestRetrieveChunksWiring:
    """retrieve_chunks should over-fetch and rerank only when enabled."""

    def _patch_collection(self, available: int):
        """A collection whose query() honors n_results, like real ChromaDB."""
        coll = MagicMock()
        coll.count.return_value = available

        def _query(query_embeddings, n_results, include):
            k = min(n_results, available)
            return {
                "documents": [[f"chunk {i}" for i in range(k)]],
                "metadatas": [[{"page_number": i, "chunk_id": f"c{i}"} for i in range(k)]],
                "distances": [[0.1 * i for i in range(k)]],
            }

        coll.query.side_effect = _query
        return coll

    def test_disabled_fetches_top_k_and_skips_rerank(self):
        coll = self._patch_collection(available=50)
        emb = MagicMock()
        emb.embed_query.return_value = [0.1] * 384
        with patch.object(vs.settings, "rerank_enabled", False), \
             patch("app.rag.vectorstore.get_embedding_model", return_value=emb), \
             patch("app.rag.vectorstore.get_or_create_collection", return_value=coll), \
             patch("app.rag.vectorstore._apply_rerank") as rerank_spy:
            out = vs.retrieve_chunks("doc", "q", top_k=5)

        rerank_spy.assert_not_called()
        assert coll.query.call_args.kwargs["n_results"] == 5
        assert len(out) == 5

    def test_enabled_overfetches_candidates_then_reranks(self):
        coll = self._patch_collection(available=50)
        emb = MagicMock()
        emb.embed_query.return_value = [0.1] * 384
        with patch.object(vs.settings, "rerank_enabled", True), \
             patch.object(vs.settings, "rerank_candidates", 20), \
             patch("app.rag.vectorstore.get_embedding_model", return_value=emb), \
             patch("app.rag.vectorstore.get_or_create_collection", return_value=coll), \
             patch("app.rag.vectorstore._apply_rerank", return_value=[]) as rerank_spy:
            vs.retrieve_chunks("doc", "q", top_k=5)

        assert coll.query.call_args.kwargs["n_results"] == 20
        rerank_spy.assert_called_once()
        assert rerank_spy.call_args.args[2] == 5  # truncates to top_k
