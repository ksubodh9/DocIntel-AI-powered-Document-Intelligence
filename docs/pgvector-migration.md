# Migrating the vector store from ChromaDB to pgvector

## Why this changed

The retriever returned **0 chunks on the live server** while working locally.
Root cause: the document rows live in durable Postgres/Supabase, but the vectors
lived in a **ChromaDB PersistentClient on the container's local disk**
(`data/vectorstore`). Managed hosts wipe that filesystem on every redeploy,
restart, or scale event. After a wipe, the DB still lists the document but its
ChromaDB collection is gone, so `retrieve_chunks` hits an empty collection,
returns `[]`, and every answer collapses to *"I could not find this information
in the document."*

Log signature of the bug:

```
[Chat] Retrieved 0 chunks in 0.68s  scores=[]
[Chat] All 0 retrieved chunks below relevance threshold (0.35) ... Returning 'not found'.
```

The fix moves vectors into the **same durable Postgres you already use for
documents**, via the pgvector extension. One source of truth, survives
redeploys automatically, and works across multiple replicas (no shared-disk
constraint).

## What changed in the code

- `app/rag/vectorstore.py` — rewritten on pgvector (raw SQL over the existing
  SQLAlchemy engine). **Public API is unchanged**: `index_chunks`,
  `retrieve_chunks`, `retrieve_chunks_multi`, `delete_collection`, plus
  `get_or_create_collection` / `get_chroma_client` compatibility shims so
  `scripts/reindex.py`, `scripts/diagnose_*.py`, and the startup stale-index
  check in `app/main.py` keep working unmodified.
- `app/config/settings.py` — added `embedding_dim` (default `384`, matching
  `BAAI/bge-small-en-v1.5`).
- Old Chroma module preserved at `app/rag/_vectorstore_chroma_legacy.py.bak`.

Cosine semantics are preserved: vectors are L2-normalized, pgvector's `<=>`
returns `1 - cosine_similarity`, so `relevance_score = 1 - distance` keeps the
same range and the `min_relevance_score = 0.35` threshold carries over unchanged.

## Storage model

A single table, one row per chunk, filtered by `doc_id` (this replaces Chroma's
one-collection-per-document scheme):

```
chunk_embeddings(
  chunk_id PK, doc_id, chunk_index, page_number, word_count,
  text, embedding vector(384), embedding_model, created_at
)
```

Indexes: btree on `doc_id` (every query filters by it) and an HNSW cosine index
on `embedding`. The table, extension, and indexes are created automatically on
first use (`CREATE EXTENSION/TABLE/INDEX IF NOT EXISTS`) — no manual DDL needed.

## Deploy steps

1. **Enable pgvector on Supabase** (one-time). Dashboard → Database →
   Extensions → enable `vector`, or run in the SQL editor:
   ```sql
   create extension if not exists vector;
   ```
   (The app also runs this on startup, but enabling it explicitly avoids a
   permissions surprise on locked-down roles.)
2. **Set `EMBEDDING_DIM`** if you are not on the default BGE model:
   `BAAI/bge-small-en-v1.5 → 384`, `text-embedding-3-small → 1536`.
3. **Deploy.** The `chunk_embeddings` table is created on the first upload or
   chat.
4. **Re-index existing documents** whose source files still exist:
   ```bash
   python scripts/reindex.py --dry-run   # preview
   python scripts/reindex.py             # rebuild all ready docs into pgvector
   ```

## Important caveat — uploaded PDFs are also on ephemeral disk

`upload_dir` (`data/uploads`) has the **same ephemeral-disk problem**. After a
redeploy the original PDFs are gone, so `scripts/reindex.py` cannot rebuild
those documents (`file missing on disk`) — users must re-upload them once.

To make the system fully durable, move raw uploads to object storage
(**Supabase Storage** or S3) and store the object key instead of a local path in
`Document.file_path`. This is a separate change from the vector store fix but is
required to survive redeploys end-to-end. Until then: after this deploy, ask
users to re-upload; new uploads embed straight into pgvector and persist.

## Local development

pgvector requires Postgres — the SQLite dev DB cannot store embeddings (the code
raises a clear error if `DATABASE_URL` is SQLite when you try to index/retrieve).
For parity, either point `DATABASE_URL` at Supabase, or run Postgres locally:

```bash
docker run -d --name pgvector -e POSTGRES_PASSWORD=pass -p 5432:5432 pgvector/pgvector:pg16
# DATABASE_URL=postgresql+psycopg2://postgres:pass@localhost:5432/postgres
```

The pytest suite is unaffected: it uses in-memory SQLite and this change adds no
ORM model to `Base.metadata`, so `create_all` on SQLite is unchanged and the
Postgres paths are never exercised by the tests.

## Rollback

Restore the previous module and redeploy:

```bash
git checkout -- app/rag/vectorstore.py app/config/settings.py
# or, without git:
cp app/rag/_vectorstore_chroma_legacy.py.bak app/rag/vectorstore.py
```

Note the original bug returns with the rollback (vectors back on ephemeral disk).

## Multi-replica note

Because vectors now live in Postgres, you can safely run more than one API
replica — an upload handled by one replica is immediately queryable from any
other. The old Chroma PersistentClient could not do this.
