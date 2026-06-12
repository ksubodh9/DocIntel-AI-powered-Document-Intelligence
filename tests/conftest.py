"""
Shared pytest fixtures for the PDF Agent test suite.

Sets up:
  * A known SUPABASE_JWT_SECRET so we can mint real HS256 tokens and exercise
    the actual auth/verification path (no dependency overrides for auth).
  * An in-memory SQLite DB shared across threads (StaticPool) so seeded rows are
    visible to FastAPI's threadpool-run sync handlers.
  * Token + document seeding helpers.
"""

import os

# IMPORTANT: set auth env BEFORE importing the app so settings/auth capture it.
TEST_JWT_SECRET = "test-jwt-secret-do-not-use-in-prod"
os.environ["SUPABASE_JWT_SECRET"] = TEST_JWT_SECRET
os.environ["ALLOW_INSECURE_AUTH"] = "false"

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import jwt as pyjwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Patch the heavy embedding model before importing the app.
_patchers = [
    patch("app.rag.embeddings.BGEEmbeddings.__init__", return_value=None),
    patch("app.rag.embeddings.BGEEmbeddings.embed_documents", return_value=[[0.1] * 384]),
    patch("app.rag.embeddings.BGEEmbeddings.embed_query", return_value=[0.1] * 384),
]
for _p in _patchers:
    _p.start()

from app.main import app                       # noqa: E402
from app.database.base import get_db, Base     # noqa: E402
from app.models.document import Document       # noqa: E402
import app.middleware.auth as auth_mod         # noqa: E402

# The auth module captures the secret at import time — make sure it's ours.
auth_mod._JWT_SECRET = TEST_JWT_SECRET

# Single shared in-memory DB (StaticPool keeps one connection across threads).
test_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
Base.metadata.create_all(bind=test_engine)


def _override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = _override_get_db


# ── Token helpers ─────────────────────────────────────────────────────────────

def make_token(
    user_id: str = "user-1",
    email: str = "user1@example.com",
    role: str | None = None,
    secret: str = TEST_JWT_SECRET,
    expired: bool = False,
) -> str:
    """Mint an HS256 JWT in the Supabase shape."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "app_metadata": {"role": role} if role else {},
        "iat": now,
        "exp": now - timedelta(hours=1) if expired else now + timedelta(hours=1),
    }
    return pyjwt.encode(payload, secret, algorithm="HS256")


def auth_header(**kwargs) -> dict:
    return {"Authorization": f"Bearer {make_token(**kwargs)}"}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def token():
    """Expose the token factory to tests."""
    return make_token


@pytest.fixture
def user_headers() -> dict:
    return auth_header(user_id="user-1")


@pytest.fixture
def admin_headers() -> dict:
    return auth_header(user_id="admin-1", email="admin@example.com", role="admin")


@pytest.fixture
def seed_document():
    """Insert a Document owned by `user_id` and return its id."""
    def _seed(user_id: str = "user-1", status: str = "ready") -> str:
        doc_id = str(uuid.uuid4())
        db = TestSessionLocal()
        try:
            db.add(Document(
                id=doc_id,
                user_id=user_id,
                filename=f"{doc_id}.pdf",
                original_filename="test.pdf",
                file_path=f"/tmp/{doc_id}.pdf",
                file_size=123,
                page_count=1,
                full_text="hello world",
                status=status,
            ))
            db.commit()
        finally:
            db.close()
        return doc_id
    return _seed
