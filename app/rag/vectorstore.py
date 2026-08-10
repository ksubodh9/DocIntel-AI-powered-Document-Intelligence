"""
pgvector-backed vector store.

Why pgvector instead of ChromaDB
--------------------------------
The previous implementation stored vectors in a ChromaDB PersistentClient on
the container's *local disk* (``data/vectorstore``). On managed hosts
(Render/Railway/etc.) that filesystem is ephemeral — it is wiped on every
redeploy, restart, or scale event. The document rows live in Postgres/Supabase
(durable), so after a restart the DB still lists a document but its ChromaDB
collection is gone. ``retrieve_chunks`` then finds an empty collection, returns
[], and every answer collapses to "not found" — exactly the live-server bug.

This module moves the vectors into the *same* durable Postgres you already use
for documents, via the pgvector extension. One source of truth, survives
redeploys automatically, and works across multiple replicas with no shared-disk
constraint.

Storage model
-------------
A single table ``chunk_embeddings``, one row per chunk, filtered by ``doc_id``
(this replaces Chroma's one-collection-per-document scheme). Columns:
  chunk_id (PK), doc_id, chunk_index, page_number, word_count, text,
  embedding vector(dim), embedding_model, created_at.

Similarity uses pgvector's cosine distance operator ``<=>``. Because the
embedding model L2-normalizes its vectors, cosine distance = 1 - cosine
similarity, so ``relevance_score = 1 - distance`` keeps the SAME semantics and
score range as the old Chroma path — the ``min_relevance_score`` threshold
(0.35) carries over unchanged.

Public interface (unchanged, drop-in for the old Chroma module):
  index_chunks(doc_id, chunks) -> str
  retrieve_chunks(doc_id, query, top_k=None) -> list[dict]
  retrieve_chunks_multi(doc_ids, query, top_k=None) -> list[dict]
  delete_collection(doc_id) -> None

Compatibility shims (so scripts/reindex.py, scripts/diagnose_*.py and the
startup stale-index check in app/main.py keep working without edits):
  get_or_create_collection(doc_id) -> _PgCollection
  get_chroma_client() -> _PgClient

Note: this store requires PostgreSQL. The local SQLite dev DB does not support
pgvector; run Postgres locally (e.g. the pgvector docker image) for parity, or
point DATABASE_URL at your Supabase instance.
"""

import logging
from functools import lru_cache
from typing import Optional

from sqlalchemy import text, bindparam

from app.config.settings import get_settings
from app.database.base import engine
from app.rag.embeddings import get_embedding_model
from app.rag.chunker import TextChunk

logger = logging.getLogger(__name__)
settings = get_settings()

_TABLE = "chunk_embeddings"
_schema_ready = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_postgres() -> bool:
    return not settings.database_url.startswith("sqlite")


def _vec_literal(vec) -> str:
    """
    Render a float vector as a pgvector text literal: '[0.1,0.2,...]'.
    Passed as a bind param and cast with CAST(:x AS vector) in SQL, so there is
    no injection surface and no dependency on a Python pgvector adapter.
    """
    return "[" + ",".join(f"{float(x):.7f}" for x in vec) + "]"


def _ensure_schema() -> None:
    """
    Create the extension, table, and indexes once per process (idempotent).
    Raises a clear error on SQLite — embeddings require Postgres/pgvector.
    """
    global _schema_ready
    if _schema_ready:
        return
    if not _is_postgres():
        raise RuntimeError(
            "The pgvector vector store requires PostgreSQL. Your DATABASE_URL is "
            "SQLite, which cannot store embeddings. Set DATABASE_URL to your "
            "Supabase/Postgres URL (run a local Postgres+pgvector for dev parity)."
        )

    dim = settings.embedding_dim
    with engine.begin() as conn:
        # Supabase ships pgvector; this enables it (no-op if already enabled).
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {_TABLE} (
                chunk_id        VARCHAR(255) PRIMARY KEY,
                doc_id          VARCHAR(36)  NOT NULL,
                chunk_index     INTEGER      NOT NULL DEFAULT 0,
                page_number     INTEGER      NOT NULL DEFAULT 1,
                word_count      INTEGER      NOT NULL DEFAULT 0,
                text            TEXT         NOT NULL,
                embedding       vector({dim}) NOT NULL,
                embedding_model VARCHAR(255) NOT NULL,
                created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
            )
        """))
        # btree on doc_id: every query filters by it, and per-doc chunk counts
        # are small so an exact scan within a document is fast.
        conn.execute(text(
            f"CREATE INDEX IF NOT EXISTS idx_{_TABLE}_doc ON {_TABLE} (doc_id)"
        ))
        # HNSW ANN index for cosine similarity across the whole table (helps the
        # multi-document path and scales as the corpus grows). Requires
        # pgvector >= 0.5; Supabase satisfies this.
        try:
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{_TABLE}_embedding
                ON {_TABLE} USING hnsw (embedding vector_cosine_ops)
            """))
        except Exception as e:  # older pgvector without HNSW — cosine still works
            logger.warning(f"[VectorStore] HNSW index not created ({e}); using exact scan.")

    _schema_ready = True
    logger.info(f"[VectorStore] pgvector schema ready (table={_TABLE}, dim={dim}).")


def _apply_rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """
    Reorder candidate chunks with the cross-encoder reranker and keep the best
    `top_k`. Each chunk keeps its original cosine `relevance_score` (so the
    downstream `min_relevance_score` filter is unchanged) and gains a
    `rerank_score`. Returns chunks ordered by `rerank_score` descending.

    If the reranker fails to load or score for any reason, we fall back to the
    cosine ordering already on the chunks — retrieval must never hard-fail just
    because reranking is unavailable.
    """
    if not chunks:
        return chunks
    try:
        from app.rag.reranker import get_reranker

        scores = get_reranker().rerank(query, [c["text"] for c in chunks])
        for c, s in zip(chunks, scores):
            c["rerank_score"] = round(float(s), 4)
        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        logger.info(
            f"[VectorStore] Reranked {len(chunks)} candidates "
            f"(top rerank_score={chunks[0]['rerank_score']:.3f}), keeping top {top_k}"
        )
    except Exception as e:
        logger.warning(f"[VectorStore] Rerank failed ({e}); falling back to cosine order")
    return chunks[:top_k]


# ── Indexing ──────────────────────────────────────────────────────────────────

def index_chunks(doc_id: str, chunks: list[TextChunk]) -> str:
    """
    Embed and upsert all chunks for a document into pgvector.
    Returns the logical collection name ("doc_{doc_id}") for backward
    compatibility with callers that persist it on Document.collection_name.
    """
    if not chunks:
        raise ValueError("No chunks to index.")

    _ensure_schema()
    embedding_model = get_embedding_model()

    texts = [chunk.text for chunk in chunks]

    # Embed in batches of 64 to avoid memory spikes (same as the Chroma path).
    batch_size = 64
    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    logger.info(f"[VectorStore] Starting embedding: {len(chunks)} chunks in {total_batches} batch(es) of {batch_size}")
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"[VectorStore] Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        all_embeddings.extend(embedding_model.embed_documents(batch))
        logger.info(f"[VectorStore] Batch {batch_num}/{total_batches} done")

    # Guard: the column is fixed-dimension. A mismatch (e.g. swapping to a
    # 1536-dim model without updating embedding_dim + re-creating the table)
    # would fail deep in the driver — fail early and legibly instead.
    got_dim = len(all_embeddings[0]) if all_embeddings else settings.embedding_dim
    if got_dim != settings.embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch: model produced {got_dim}-dim vectors "
            f"but settings.embedding_dim={settings.embedding_dim}. Update EMBEDDING_DIM "
            f"to {got_dim}, drop/recreate the {_TABLE} table, and re-index."
        )

    rows = [
        {
            "chunk_id": chunk.chunk_id,
            "doc_id": doc_id,
            "chunk_index": chunk.chunk_index,
            "page_number": chunk.page_number,
            "word_count": chunk.word_count,
            "text": chunk.text,
            "embedding": _vec_literal(emb),
            "embedding_model": settings.embedding_model,
        }
        for chunk, emb in zip(chunks, all_embeddings)
    ]

    stmt = text(f"""
        INSERT INTO {_TABLE}
            (chunk_id, doc_id, chunk_index, page_number, word_count, text, embedding, embedding_model)
        VALUES
            (:chunk_id, :doc_id, :chunk_index, :page_number, :word_count, :text,
             CAST(:embedding AS vector), :embedding_model)
        ON CONFLICT (chunk_id) DO UPDATE SET
            doc_id          = EXCLUDED.doc_id,
            chunk_index     = EXCLUDED.chunk_index,
            page_number     = EXCLUDED.page_number,
            word_count      = EXCLUDED.word_count,
            text            = EXCLUDED.text,
            embedding       = EXCLUDED.embedding,
            embedding_model = EXCLUDED.embedding_model
    """)

    logger.info("[VectorStore] Writing embeddings to pgvector...")
    with engine.begin() as conn:
        conn.execute(stmt, rows)  # executemany over the row dicts
    logger.info(f"[VectorStore] Indexed {len(chunks)} chunks for doc_{doc_id}")
    return f"doc_{doc_id}"


# ── Retrieval ─────────────────────────────────────────────────────────────────

def _count_and_models(doc_id: str) -> tuple[int, set]:
    """Return (row_count, {embedding_model,...}) for a document."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"SELECT embedding_model FROM {_TABLE} WHERE doc_id = :d"),
            {"d": doc_id},
        ).scalars().all()
    return len(rows), set(rows)


def retrieve_chunks(
    doc_id: str,
    query: str,
    top_k: Optional[int] = None,
) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a query within one document.
    Returns list of dicts: {text, page_number, chunk_id, distance, relevance_score}
    (plus rerank_score when reranking is enabled).
    """
    top_k = top_k or settings.top_k_retrieval
    _ensure_schema()
    embedding_model = get_embedding_model()

    # Guard: the document must have indexed vectors.
    count, models = _count_and_models(doc_id)
    if count == 0:
        return []

    # Guard: vectors must have been produced by the model we're querying with.
    # A mismatch means cosine scores are noise and every answer collapses to
    # "not found" — warn loudly so it's diagnosable (re-index with reindex.py).
    stale = {m for m in models if m and m != settings.embedding_model}
    if stale:
        logger.warning(
            f"[VectorStore] STALE INDEX for doc {doc_id}: vectors were built with "
            f"{sorted(stale)} but the current model is '{settings.embedding_model}'. "
            f"Retrieval scores will be unreliable — re-index with scripts/reindex.py."
        )

    # When reranking is on, over-fetch a larger candidate pool so the
    # cross-encoder has more to reorder, then truncate back to top_k below.
    n_fetch = settings.rerank_candidates if settings.rerank_enabled else top_k

    query_vec = _vec_literal(embedding_model.embed_query(query))
    sql = text(f"""
        SELECT chunk_id, page_number, text,
               (embedding <=> CAST(:q AS vector)) AS distance
        FROM {_TABLE}
        WHERE doc_id = :d
        ORDER BY distance ASC
        LIMIT :k
    """)
    with engine.connect() as conn:
        result = conn.execute(sql, {"q": query_vec, "d": doc_id, "k": int(n_fetch)}).mappings().all()

    chunks = []
    for r in result:
        distance = float(r["distance"])
        chunks.append(
            {
                "text": r["text"],
                "page_number": r["page_number"] or 1,
                "chunk_id": r["chunk_id"] or "",
                "distance": round(distance, 4),
                "relevance_score": round(1 - distance, 4),  # cosine similarity
            }
        )

    logger.info(
        f"[VectorStore] Retrieved {len(chunks)} chunks for query "
        f"(top score={chunks[0]['relevance_score'] if chunks else 0:.3f})"
    )

    if settings.rerank_enabled:
        return _apply_rerank(query, chunks, top_k)
    return chunks


def retrieve_chunks_multi(
    doc_ids: list[str],
    query: str,
    top_k: int | None = None,
) -> list[dict]:
    """
    Retrieve chunks across multiple documents in a single query, merge and
    re-rank by relevance. Each result includes doc_id so the caller knows which
    document it came from.
    """
    top_k = top_k or settings.top_k_retrieval
    if not doc_ids:
        return []

    _ensure_schema()
    embedding_model = get_embedding_model()

    n_fetch = settings.rerank_candidates if settings.rerank_enabled else top_k
    # Over-fetch across the whole set so rerank/cosine can pick the global best.
    limit = int(n_fetch) * max(len(doc_ids), 1)

    query_vec = _vec_literal(embedding_model.embed_query(query))
    sql = text(f"""
        SELECT chunk_id, doc_id, page_number, text,
               (embedding <=> CAST(:q AS vector)) AS distance
        FROM {_TABLE}
        WHERE doc_id IN :ids
        ORDER BY distance ASC
        LIMIT :k
    """).bindparams(bindparam("ids", expanding=True))

    try:
        with engine.connect() as conn:
            result = conn.execute(
                sql, {"q": query_vec, "ids": list(doc_ids), "k": limit}
            ).mappings().all()
    except Exception as e:
        logger.warning(f"[VectorStore] Multi-doc retrieval failed: {e}")
        return []

    all_chunks = [
        {
            "text": r["text"],
            "doc_id": r["doc_id"],
            "page_number": r["page_number"] or 1,
            "chunk_id": r["chunk_id"] or "",
            "distance": round(float(r["distance"]), 4),
            "relevance_score": round(1 - float(r["distance"]), 4),
        }
        for r in result
    ]

    if settings.rerank_enabled:
        return _apply_rerank(query, all_chunks, top_k)
    all_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
    return all_chunks[:top_k]


# ── Deletion ──────────────────────────────────────────────────────────────────

def delete_collection(doc_id: str) -> None:
    """Delete all of a document's embeddings (called when the document is deleted)."""
    if not _is_postgres():
        return
    try:
        _ensure_schema()
        with engine.begin() as conn:
            conn.execute(text(f"DELETE FROM {_TABLE} WHERE doc_id = :d"), {"d": doc_id})
    except Exception as e:
        logger.warning(f"[VectorStore] delete_collection({doc_id}) failed: {e}")


# ── Backward-compatibility shims ──────────────────────────────────────────────
# The old Chroma module exposed a client and per-document "collection" objects.
# reindex.py, the diagnose_* scripts, and the startup stale-index check in
# app/main.py still call these. We emulate the tiny surface they rely on
# (count() and metadata) so those callers keep working unmodified.

class _PgCollection:
    """Emulates the subset of a ChromaDB Collection our callers use."""

    def __init__(self, doc_id: str):
        self.doc_id = doc_id

    def count(self) -> int:
        try:
            with engine.connect() as conn:
                return int(conn.execute(
                    text(f"SELECT count(*) FROM {_TABLE} WHERE doc_id = :d"),
                    {"d": self.doc_id},
                ).scalar() or 0)
        except Exception:
            return 0

    @property
    def metadata(self) -> dict:
        model = None
        try:
            with engine.connect() as conn:
                model = conn.execute(
                    text(f"SELECT embedding_model FROM {_TABLE} WHERE doc_id = :d LIMIT 1"),
                    {"d": self.doc_id},
                ).scalar()
        except Exception:
            pass
        return {
            "doc_id": self.doc_id,
            "embedding_model": model or settings.embedding_model,
            "hnsw:space": "cosine",
        }


class _PgClient:
    """Emulates the subset of a ChromaDB PersistentClient our callers use."""

    @staticmethod
    def _doc_id(name: str) -> str:
        return name[4:] if name.startswith("doc_") else name

    def get_collection(self, name: str) -> _PgCollection:
        col = _PgCollection(self._doc_id(name))
        if col.count() == 0:
            # Mirror Chroma raising when a collection doesn't exist, so callers'
            # try/except (e.g. the startup stale check) skip cleanly.
            raise ValueError(f"No embeddings for collection {name!r}")
        return col

    def get_or_create_collection(self, name: str, **_kwargs) -> _PgCollection:
        return _PgCollection(self._doc_id(name))

    def delete_collection(self, name: str) -> None:
        delete_collection(self._doc_id(name))


@lru_cache(maxsize=1)
def get_chroma_client() -> _PgClient:
    """Singleton compatibility client (name kept for drop-in callers)."""
    return _PgClient()


def get_or_create_collection(doc_id: str) -> _PgCollection:
    """
    Compatibility shim. pgvector needs no per-document collection object; this
    returns a lightweight handle exposing count()/metadata. Ensures the schema
    exists so reindex.py can call it before index_chunks.
    """
    try:
        _ensure_schema()
    except Exception as e:
        logger.warning(f"[VectorStore] get_or_create_collection: schema not ready ({e})")
    return _PgCollection(doc_id)
