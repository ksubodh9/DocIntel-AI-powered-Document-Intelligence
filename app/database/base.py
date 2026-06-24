"""
SQLAlchemy engine, session factory, and declarative base.
Supports both SQLite (local dev) and PostgreSQL (Supabase production).
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from typing import Generator

from app.config.settings import get_settings

settings = get_settings()
_is_sqlite = settings.database_url.startswith("sqlite")


class Base(DeclarativeBase):
    pass


if _is_sqlite:
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False, "timeout": 30},
        echo=settings.debug,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _record):
        """
        MEMORY journal mode: SQLite never writes/deletes journal files to disk.
        Eliminates SQLITE_IOERR_DELETE (code 2570) on Docker WSL2 bind-mounts.
        """
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=MEMORY")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

else:
    # PostgreSQL (Supabase) — use connection pooling
    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,       # reconnect on stale connections
        pool_size=5,
        max_overflow=10,
        echo=settings.debug,
    )


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables and run lightweight schema migrations."""
    from app.models import document, usage, user_credential   # noqa: F401 — registers models
    Base.metadata.create_all(bind=engine)
    _migrate_schema()


def _migrate_schema() -> None:
    """
    Add any columns that exist in the ORM models but are missing from the
    physical tables.  Handles databases created before columns were added
    (e.g. an old SQLite file on a cloud deployment).

    Safe to call on every startup — it is a no-op when the schema is current.
    Works for SQLite and PostgreSQL.
    """
    import logging
    import sqlalchemy as sa

    log = logging.getLogger(__name__)

    # Map each table → list of (column_name, column_definition_sql)
    # These are every column added after the initial schema was deployed.
    MIGRATIONS: dict[str, list[tuple[str, str]]] = {
        "documents": [
            ("user_id",                    "VARCHAR(36)"),
            ("original_filename",          "VARCHAR(255)"),
            ("full_text",                  "TEXT"),
            ("error_message",              "TEXT"),
            ("doc_metadata",               "JSON"),
            ("has_tables",                 "BOOLEAN DEFAULT 0"),
            ("table_count",               "INTEGER DEFAULT 0"),
            ("tables",                     "JSON"),
            ("collection_name",            "VARCHAR(255)"),
            ("updated_at",                 "DATETIME"),
        ],
        "user_credentials": [],
        "usage_events": [
            # usage_events was added later; create_all handles the table itself,
            # but list any future columns here if needed.
        ],
    }

    with engine.connect() as conn:
        for table_name, columns in MIGRATIONS.items():
            if not columns:
                continue
            # Get existing columns
            try:
                result = conn.execute(sa.text(f"PRAGMA table_info({table_name})"))
                existing = {row[1] for row in result.fetchall()}  # row[1] = column name
            except Exception:
                # PostgreSQL: use information_schema instead
                try:
                    result = conn.execute(sa.text(
                        "SELECT column_name FROM information_schema.columns "
                        f"WHERE table_name = '{table_name}'"
                    ))
                    existing = {row[0] for row in result.fetchall()}
                except Exception as e:
                    log.warning(f"Could not inspect {table_name}: {e}")
                    continue

            for col_name, col_def in columns:
                if col_name not in existing:
                    try:
                        conn.execute(sa.text(
                            f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_def}"
                        ))
                        conn.commit()
                        log.info(f"Schema migration: added {table_name}.{col_name}")
                    except Exception as e:
                        log.warning(f"Could not add {table_name}.{col_name}: {e}")


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency — yields a DB session and closes on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
