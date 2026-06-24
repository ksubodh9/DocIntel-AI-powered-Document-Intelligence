"""
Diagnose why a document returns 0 chunks on chat.

Run from the backend environment (same .env / same data dir as the API):
    python -m scripts.diagnose_retrieval <doc_id>
    python -m scripts.diagnose_retrieval            # lists all collections

It cross-checks three things:
  1. The document row in SQLite (status, page_count, collection_name).
  2. The ChromaDB collection for that doc and how many vectors it holds.
  3. A live retrieval for a trivial query, to see real scores.

If the collection count is 0, the document was never indexed (or the
vectorstore was wiped) — that's why chat says "not found". Re-index by
re-uploading, or restore the persistent volume.
"""

import sys
from app.config.settings import get_settings
from app.rag.vectorstore import get_chroma_client, get_or_create_collection, retrieve_chunks

settings = get_settings()


def list_all():
    client = get_chroma_client()
    cols = client.list_collections()
    print(f"Vectorstore dir : {settings.vectorstore_dir}")
    print(f"Collections     : {len(cols)}")
    for c in cols:
        try:
            print(f"  - {c.name:50s} count={c.count()}")
        except Exception as e:
            print(f"  - {c.name:50s} (count failed: {e})")


def inspect_doc(doc_id: str):
    print(f"=== Document {doc_id} ===\n")

    # 1. DB row
    try:
        from app.database.base import SessionLocal
        from app.models.document import Document
        db = SessionLocal()
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            print("DB: no document row with that id.")
        else:
            print("DB row:")
            print(f"  status          = {doc.status}")
            print(f"  page_count      = {doc.page_count}")
            print(f"  collection_name = {getattr(doc, 'collection_name', None)}")
            print(f"  full_text chars = {len(doc.full_text or '')}")
            print(f"  error_message   = {doc.error_message}")
        db.close()
    except Exception as e:
        print(f"DB inspect failed: {e}")

    # 2. Chroma collection
    print("\nChromaDB:")
    try:
        col = get_or_create_collection(doc_id)
        cnt = col.count()
        print(f"  collection 'doc_{doc_id}' count = {cnt}")
        if cnt == 0:
            print("  >>> EMPTY. The document has no vectors indexed.")
            print("  >>> This is why chat returns 0 chunks. Re-index the document.")
            return
        peek = col.peek(2)
        print(f"  sample metadatas = {peek.get('metadatas')}")
    except Exception as e:
        print(f"  collection inspect failed: {e}")
        return

    # 3. Live retrieval (no threshold applied here — we want raw scores)
    print("\nLive retrieval (raw scores, threshold NOT applied):")
    for q in ["summary", "the main topic of this document"]:
        chunks = retrieve_chunks(doc_id, q, top_k=5)
        scores = [c["relevance_score"] for c in chunks]
        print(f"  query={q!r:40s} -> {len(chunks)} chunks, scores={scores}")
    print(f"\nmin_relevance_score threshold = {settings.min_relevance_score}")
    print("If scores are all below that, it's a real relevance/threshold case.")
    print("If you get chunks here but chat says 'not found', the threshold is too high.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        list_all()
    else:
        inspect_doc(sys.argv[1])
