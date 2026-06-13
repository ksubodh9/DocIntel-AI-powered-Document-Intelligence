"""
One-time data migration: local SQLite  →  Supabase Postgres.

Copies every row from the local SQLite database into the Postgres database
pointed to by DATABASE_URL, table by table, in FK-safe order. Uses the app's
own SQLAlchemy models so column mapping is exact.

Usage
-----
1. Make sure DATABASE_URL in your .env points at Supabase Postgres
   (the target). Tables are created automatically if missing.
2. Run from the project root:

       python -m scripts.migrate_sqlite_to_supabase

   Optionally point at a different source SQLite file:

       python -m scripts.migrate_sqlite_to_supabase --sqlite data/pdf_agent.db

Notes
-----
- Idempotent-ish: rows whose primary key already exists in the target are
  skipped (so re-running won't duplicate). It does NOT update changed rows.
- This copies application data only. It does NOT touch Supabase `auth.users`
  (that's managed by Supabase Auth, not this app).
"""

from __future__ import annotations

import argparse
import os
import sys

from sqlalchemy import create_engine, inspect as sa_inspect
from sqlalchemy.orm import sessionmaker

# Import models so they register on Base.metadata, and so we can iterate them.
from app.database.base import Base, engine as target_engine, init_db
from app.models.document import Document, ChatMessage
from app.models.usage import UsageEvent
from app.models.feedback import Feedback

# FK-safe order: parents before children.
MODELS = [Document, ChatMessage, UsageEvent, Feedback]


def _column_names(model) -> list[str]:
    return [c.key for c in sa_inspect(model).mapper.column_attrs]


def migrate(sqlite_path: str) -> None:
    if not os.path.exists(sqlite_path):
        sys.exit(f"❌ Source SQLite file not found: {sqlite_path}")

    target_url = str(target_engine.url)
    if target_url.startswith("sqlite"):
        sys.exit(
            "❌ DATABASE_URL still points at SQLite — nothing to migrate INTO.\n"
            "   Set DATABASE_URL to your Supabase Postgres connection string first."
        )

    print(f"Source : sqlite:///{sqlite_path}")
    print(f"Target : {target_engine.url.render_as_string(hide_password=True)}\n")

    # Ensure the target schema exists.
    print("Ensuring target tables exist (init_db)…")
    init_db()

    source_engine = create_engine(f"sqlite:///{sqlite_path}")
    SourceSession = sessionmaker(bind=source_engine)
    TargetSession = sessionmaker(bind=target_engine)

    src = SourceSession()
    dst = TargetSession()

    total_copied = 0
    try:
        for model in MODELS:
            table = model.__tablename__
            cols = _column_names(model)

            # Skip cleanly if the source DB doesn't have this table yet.
            if not sa_inspect(source_engine).has_table(table):
                print(f"• {table:<14} — not in source, skipping")
                continue

            existing_ids = {row[0] for row in dst.query(model.id).all()}
            rows = src.query(model).all()

            copied = 0
            for row in rows:
                if row.id in existing_ids:
                    continue
                dst.add(model(**{c: getattr(row, c) for c in cols}))
                copied += 1

            dst.commit()
            total_copied += copied
            print(f"• {table:<14} — {copied} copied / {len(rows)} in source "
                  f"({len(existing_ids)} already present)")

        print(f"\n✅ Done. {total_copied} new row(s) migrated to Supabase.")
    except Exception:
        dst.rollback()
        raise
    finally:
        src.close()
        dst.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy app data from SQLite to Supabase Postgres.")
    parser.add_argument(
        "--sqlite",
        default="data/pdf_agent.db",
        help="Path to the source SQLite file (default: data/pdf_agent.db)",
    )
    args = parser.parse_args()
    migrate(args.sqlite)
