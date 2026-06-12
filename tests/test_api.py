"""
Integration tests for FastAPI endpoints (auth-aware).

Shared setup — app import, in-memory DB override, and auth/token helpers —
lives in conftest.py. Endpoints now require a valid JWT, so requests pass the
`user_headers` fixture where authentication is expected to succeed.
"""

import app.config.settings as cfg

API = "/api/v1"


class TestHealth:
    def test_health_check(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestUpload:
    def test_upload_requires_auth(self, client):
        r = client.post(
            f"{API}/upload",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert r.status_code == 401

    def test_upload_non_pdf_extension_rejected(self, client, user_headers):
        r = client.post(
            f"{API}/upload",
            headers=user_headers,
            files={"file": ("test.exe", b"hello", "application/octet-stream")},
        )
        assert r.status_code == 400

    def test_upload_empty_file_rejected(self, client, user_headers, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg.get_settings(), "upload_dir", tmp_path)
        r = client.post(
            f"{API}/upload",
            headers=user_headers,
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        # Either rejected up front (400/422) or accepted and marked errored.
        assert r.status_code in (400, 422) or (
            r.status_code == 201 and r.json().get("status") == "error"
        )


class TestDocumentNotFound:
    def test_get_nonexistent_document(self, client, user_headers):
        r = client.get(f"{API}/document/nonexistent-id", headers=user_headers)
        assert r.status_code == 404

    def test_classify_nonexistent(self, client, user_headers):
        r = client.post(f"{API}/classify/nonexistent-id", headers=user_headers)
        assert r.status_code == 404

    def test_chat_nonexistent(self, client, user_headers):
        r = client.post(
            f"{API}/chat",
            headers=user_headers,
            json={"document_id": "nonexistent", "message": "hello"},
        )
        assert r.status_code == 404

    def test_questions_nonexistent(self, client, user_headers):
        r = client.get(f"{API}/questions/nonexistent-id", headers=user_headers)
        assert r.status_code == 404


class TestChatValidation:
    def test_chat_empty_message_rejected(self, client, user_headers):
        r = client.post(
            f"{API}/chat",
            headers=user_headers,
            json={"document_id": "some-id", "message": ""},
        )
        assert r.status_code == 422

    def test_chat_missing_doc_id_rejected(self, client, user_headers):
        r = client.post(f"{API}/chat", headers=user_headers, json={"message": "hello"})
        assert r.status_code == 422
