"""
SQLAlchemy declarative base and session factory.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from typing import Generator

from app.config.settings import get_settings

settings = get_settings()


class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.database_url,
    # check_same_thread=False: SQLite default forbids cross-thread use; FastAPI
    # runs sync handlers in a threadpool so we need this.
    # timeout=30: wait up to 30s for a write lock before failing.
    connect_args={"check_same_thread": False, "timeout": 30},
    echo=settings.debug,
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_conn, _record):
    """
    Use MEMORY journal mode so SQLite never writes or deletes journal files on disk.
    This eliminates SQLITE_IOERR_DELETE (code 2570) errors that occur when Docker
    volume filesystems (WSL2 bind-mounts or named volumes) can't handle SQLite's
    journal file lifecycle. The journal is kept in RAM — safe for normal operation,
    the only risk is data loss if the process is killed mid-transaction (acceptable
    for a development/demo environment).
    """
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=MEMORY")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables. Call once at startup."""
    from app.models import document  # noqa: F401
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency - yields a DB session and ensures it closes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
