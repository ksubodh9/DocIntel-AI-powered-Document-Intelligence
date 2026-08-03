"""
Re-index documents into ChromaDB with the CURRENT embedding pipeline.

Why you need this
-----------------
ChromaDB stores the vectors that existed at upload time. If the embedding model
changed after a document was indexed (e.g. the sentence-transformers -> fastembed
migration), the stored vectors and today's query vectors live in different spaces,
cosine similarity collapses, and every chat answer becomes
"I could not find this information in the document."

This script re-extracts each document from its file on disk, re-chunks it, drops
the old collection, and re-embeds everything with the model in your current .env.
It is idempotent — safe to run repeatedly.

Usage (from the project root, with your venv active so the model loads):

    python scripts/reindex.py --dry-run          # show what would be re-indexed
    python scripts/reindex.py                     # re-index every ready document
    python scripts/reindex.py --doc-id <uuid>     # re-index a single document
    python scripts/reindex.py --stale-only        # only collections built by a different model
    python scripts/reindex.py --force             # also re-index docs whose files are missing (skips them)
"""
import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/reindex.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import get_settings
from app.database.base import SessionLocal
from app.models.document import Document
from app.rag.chunker import chunk_pages
from app.rag.vectorstore import (
    index_chunks,
    delete_collection,
    get_or_create_collection,
    get_chroma_client,
)
from app.services.document_service import _make_table_chunks
from app.utils.doc_utils import extract_document

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("reindex")
settings = get_settings()


def _collection_model(doc_id: str) -> str | None:
    """Embedding model stamped on an existing collection, or None if not stamped."""
    try:
        col = get_chroma_client().get_collection(f"doc_{doc_id}")
        return (col.metadata or {}).get("embedding_model")
    except Exception:
        return None


def reindex_one(db, doc: Document) -> tuple[bool, str]:
    """Re-extract, re-chunk, and re-embed a single document. Returns (ok, message)."""
    path = Path(doc.file_path)
    if not path.exists():
        return False, f"file missing on disk: {doc.file_path}"

    content = extract_document(path, doc.original_filename)
    if not content.full_text.strip():
        return False, "no extractable text"

    chunks = chunk_pages(content.pages, doc.id)
    if content.tables:
        chunks.extend(_make_table_chunks(content.tables, doc.id, len(chunks)))
    if not chunks:
        return False, "produced 0 chunks"

    # Drop the old (possibly stale) vectors, then rebuild with the current model.
    delete_collection(doc.id)
    get_or_create_collection(doc.id)  # recreate with current embedding_model stamp
    collection_name = index_chunks(doc.id, chunks)

    doc.collection_name = collection_name
    db.commit()
    return True, f"{len(chunks)} chunks -> {collection_name}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doc-id", help="Re-index only this document id.")
    ap.add_argument("--dry-run", action="store_true", help="List work without doing it.")
    ap.add_argument("--stale-only", action="store_true",
                    help="Only re-index collections stamped with a different model.")
    args = ap.parse_args()

    db = SessionLocal()
    try:
        q = db.query(Document).filter(Document.status == "ready")
        if args.doc_id:
            q = q.filter(Document.id == args.doc_id)
        docs = q.order_by(Document.created_at).all()

        if args.stale_only:
            docs = [d for d in docs
                    if _collection_model(d.id) not in (None, settings.embedding_model)] \
                   + [d for d in docs if _collection_model(d.id) is None]
            # (missing stamp == pre-migration == treat as stale)

        logger.info(f"Current embedding model: {settings.embedding_model}")
        logger.info(f"Documents to process: {len(docs)}")

        ok = failed = 0
        for i, doc in enumerate(docs, 1):
            stamp = _collection_model(doc.id) or "<none>"
            prefix = f"[{i}/{len(docs)}] {doc.id}  ({doc.original_filename})  indexed_by={stamp}"
            if args.dry_run:
                logger.info(f"{prefix}  -> WOULD RE-INDEX")
                continue
            try:
                success, msg = reindex_one(db, doc)
                if success:
                    ok += 1
                    logger.info(f"{prefix}  -> OK: {msg}")
                else:
                    failed += 1
                    logger.warning(f"{prefix}  -> SKIPPED: {msg}")
            except Exception as e:
                failed += 1
                logger.exception(f"{prefix}  -> ERROR: {e}")

        if not args.dry_run:
            logger.info(f"Done. Re-indexed {ok}, skipped/failed {failed}.")
        return 0 if failed == 0 else 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
