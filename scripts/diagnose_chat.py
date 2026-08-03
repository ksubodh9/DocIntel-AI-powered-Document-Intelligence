"""
Diagnose the "I could not find this information in the document." problem.

Run from the project root (with your venv active, so the BGE model loads):

    python scripts/diagnose_chat.py <doc_id> "your question here"

If you omit args it picks the collection with the most chunks and asks a
generic question. It prints the raw cosine relevance scores and shows exactly
which layer is dropping the answer.
"""
import sys
from app.config.settings import get_settings
from app.rag.vectorstore import get_chroma_client, retrieve_chunks

settings = get_settings()


def pick_default_doc_id() -> str:
    client = get_chroma_client()
    best, best_n = None, -1
    for col in client.list_collections():
        n = col.count()
        if n > best_n:
            best, best_n = col, n
    if not best:
        raise SystemExit("No collections found in the vector store.")
    return best.name.replace("doc_", "")


def main():
    doc_id = sys.argv[1] if len(sys.argv) > 1 else pick_default_doc_id()
    query = sys.argv[2] if len(sys.argv) > 2 else "What is this document about?"

    print(f"embedding_model     = {settings.embedding_model}")
    print(f"min_relevance_score = {settings.min_relevance_score}")
    print(f"top_k_retrieval     = {settings.top_k_retrieval}")
    print(f"rerank_enabled      = {settings.rerank_enabled}")
    print(f"doc_id              = {doc_id}")
    print(f"query               = {query!r}\n")

    chunks = retrieve_chunks(doc_id, query, top_k=settings.top_k_retrieval)
    if not chunks:
        print(">>> retrieve_chunks returned NOTHING. Collection is empty or the "
              "doc_id has no vectors. That is why chat says 'not found'.")
        return

    print("Retrieved chunks (score = cosine similarity, 1.0 = identical):")
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. score={c['relevance_score']:.3f}  page={c['page_number']}  "
              f"{c['text'][:70]!r}")

    top = max(c["relevance_score"] for c in chunks)
    passing = [c for c in chunks if c["relevance_score"] >= settings.min_relevance_score]
    print(f"\ntop score = {top:.3f}   threshold = {settings.min_relevance_score}")
    if not passing:
        print(">>> ALL chunks are BELOW the threshold -> chat returns 'not found' "
              "WITHOUT ever calling the LLM.")
        print(">>> If the top score is low (< ~0.4) even for an obviously relevant "
              "query, your stored vectors and query vectors are in different "
              "embedding spaces (stale index) OR the threshold is too high.")
    else:
        print(f">>> {len(passing)} chunk(s) pass the threshold. Retrieval is FINE; "
              "the problem is in the LLM step (check provider/key or the QA prompt).")


if __name__ == "__main__":
    main()
