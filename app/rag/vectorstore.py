import logging
"""
ChromaDB vector store wrapper.
One collection per document (named by document_id) so documents stay isolated.

Metadata stored per chunk:
  - doc_id, chunk_id, page_number, chunk_index, word_count
This metadata is returned in query results and used for citation generation.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import Optional
from functools import lru_cache

from app.config.settings import get_settings

logger = logging.getLogger(__name__)
from app.rag.embeddings import get_embedding_model
from app.rag.chunker import TextChunk

settings = get_settings()


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    """Singleton ChromaDB client backed by disk."""
    return chromadb.PersistentClient(
        path=str(settings.vectorstore_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_or_create_collection(doc_id: str) -> chromadb.Collection:
    """Get or create a ChromaDB collection for a document."""
    client = get_chroma_client()
    collection_name = f"doc_{doc_id}"
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"doc_id": doc_id, "hnsw:space": "cosine"},
    )


def delete_collection(doc_id: str) -> None:
    """Delete a document's collection (called when document is deleted)."""
    client = get_chroma_client()
    collection_name = f"doc_{doc_id}"
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # Collection may not exist


def index_chunks(doc_id: str, chunks: list[TextChunk]) -> str:
    """
    Embed and store all chunks in the document's ChromaDB collection.
    Returns the collection name.
    """
    if not chunks:
        raise ValueError("No chunks to index.")

    embedding_model = get_embedding_model()
    collection = get_or_create_collection(doc_id)

    texts = [chunk.text for chunk in chunks]
    ids = [chunk.chunk_id for chunk in chunks]
    metadatas = [
        {
            "doc_id": doc_id,
            "chunk_id": chunk.chunk_id,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "word_count": chunk.word_count,
        }
        for chunk in chunks
    ]

    # Embed in batches of 64 to avoid memory spikes
    batch_size = 64
    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    logger.info(f"[VectorStore] Starting embedding: {len(chunks)} chunks in {total_batches} batch(es) of {batch_size}")
    logger.info("[VectorStore] Loading BGE embedding model (first run: ~30s, cached after)...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"[VectorStore] Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        all_embeddings.extend(embedding_model.embed_documents(batch))
        logger.info(f"[VectorStore] Batch {batch_num}/{total_batches} done")

    logger.info(f"[VectorStore] All embeddings computed, writing to ChromaDB...")
    collection.upsert(ids=ids, embeddings=all_embeddings, documents=texts, metadatas=metadatas)
    logger.info(f"[VectorStore] Indexed {len(chunks)} chunks into collection doc_{doc_id}")
    return f"doc_{doc_id}"


def retrieve_chunks(
    doc_id: str,
    query: str,
    top_k: Optional[int] = None,
) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a query.
    Returns list of dicts: {text, page_number, chunk_id, distance}
    """
    top_k = top_k or settings.top_k_retrieval
    embedding_model = get_embedding_model()
    collection = get_or_create_collection(doc_id)

    # Guard: collection must have documents
    if collection.count() == 0:
        return []

    query_embedding = embedding_model.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for text, meta, distance in zip(docs, metas, distances):
        chunks.append(
            {
                "text": text,
                "page_number": meta.get("page_number", 1),
                "chunk_id": meta.get("chunk_id", ""),
                "distance": round(distance, 4),
                "relevance_score": round(1 - distance, 4),  # cosine similarity
            }
        )

    # Sort by relevance (ascending distance = more similar)
    chunks.sort(key=lambda x: x["distance"])
    logger.info(f"[VectorStore] Retrieved {len(chunks)} chunks for query (top score={chunks[0]['relevance_score'] if chunks else 0:.3f})")
    return chunks
