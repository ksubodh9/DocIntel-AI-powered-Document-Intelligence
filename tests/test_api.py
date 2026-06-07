"""
Integration tests for FastAPI endpoints.
Uses TestClient + in-memory SQLite + mocked LLM/embeddings.
"""

import pytest
import io
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Patch heavy dependencies before importing the app
with patch("app.rag.embeddings.BGEEmbeddings.__init__", return_value=None), \
     patch("app.rag.embeddings.BGEEmbeddings.embed_documents", return_value=[[0.1] * 384]), \
     patch("app.rag.embeddings.BGEEmbeddings.embed_query", return_value=[0.1] * 384):
    from app.main import app
    from app.database.base import get_db, Base, engine

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Test DB (in-memory SQLite)
TEST_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

# Create test tables
Base.metadata.create_all(bind=test_engine)

client = TestClient(app)


class TestHealth:
    def test_health_check(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestUpload:
    def _make_pdf_bytes(self) -> bytes:
        """Minimal valid PDF bytes."""
        return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\nxref\n0 0\ntrailer\n<< >>\nstartxref\n0\n%%EOF"

    def test_upload_non_pdf_rejected(self):
        r = client.post(
            "/api/v1/upload",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert r.status_code == 400

    def test_upload_empty_file_rejected(self):
        r = client.post(
            "/api/v1/upload",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        assert r.status_code in (400, 422)


class TestDocumentNotFound:
    def test_get_nonexistent_document(self):
        r = client.get("/api/v1/document/nonexistent-id")
        assert r.status_code == 404

    def test_classify_nonexistent(self):
        r = client.post("/api/v1/classify/nonexistent-id")
        assert r.status_code == 404

    def test_chat_nonexistent(self):
        r = client.post(
            "/api/v1/chat",
            json={"document_id": "nonexistent", "message": "hello"},
        )
        assert r.status_code == 404

    def test_questions_nonexistent(self):
        r = client.get("/api/v1/questions/nonexistent-id")
        assert r.status_code == 404


class TestChatValidation:
    def test_chat_empty_message_rejected(self):
        r = client.post(
            "/api/v1/chat",
            json={"document_id": "some-id", "message": ""},
        )
        assert r.status_code == 422

    def test_chat_missing_doc_id_rejected(self):
        r = client.post("/api/v1/chat", json={"message": "hello"})
        assert r.status_code == 422
