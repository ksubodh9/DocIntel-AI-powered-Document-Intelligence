"""
Security regression tests.

Covers the hardening added to the backend:
  * Authentication is required and signatures are verified (no unauth access,
    forged/expired tokens rejected).
  * Per-user ownership (IDOR) — a user cannot read/chat/delete another's docs.
  * Admin endpoints require a verified admin claim.
  * Upload guards — oversized files and content/extension mismatches rejected.
  * Chat question-length cap.

These exercise the REAL auth path using HS256 tokens signed with the test
secret configured in conftest.py.
"""

import io

import app.config.settings as cfg
from tests.conftest import make_token

API = "/api/v1"


def _pdf_bytes() -> bytes:
    return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< >>\n%%EOF"


# ── Authentication required ───────────────────────────────────────────────────

class TestAuthenticationRequired:
    def test_list_documents_requires_auth(self, client):
        assert client.get(f"{API}/documents").status_code == 401

    def test_get_document_requires_auth(self, client):
        assert client.get(f"{API}/document/whatever").status_code == 401

    def test_chat_requires_auth(self, client):
        r = client.post(f"{API}/chat", json={"document_id": "x", "message": "hi"})
        assert r.status_code == 401

    def test_delete_requires_auth(self, client):
        assert client.delete(f"{API}/document/whatever").status_code == 401

    def test_malformed_auth_header_rejected(self, client):
        r = client.get(f"{API}/documents", headers={"Authorization": "Token abc"})
        assert r.status_code == 401


# ── Token verification ────────────────────────────────────────────────────────

class TestTokenVerification:
    def test_token_signed_with_wrong_secret_rejected(self, client):
        bad = make_token(user_id="user-1", secret="attacker-secret")
        r = client.get(f"{API}/documents", headers={"Authorization": f"Bearer {bad}"})
        assert r.status_code == 401

    def test_expired_token_rejected(self, client):
        tok = make_token(user_id="user-1", expired=True)
        r = client.get(f"{API}/documents", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 401

    def test_valid_token_accepted(self, client, user_headers):
        assert client.get(f"{API}/documents", headers=user_headers).status_code == 200


# ── Ownership / IDOR ──────────────────────────────────────────────────────────

class TestDocumentOwnership:
    def test_owner_can_read_own_document(self, client, user_headers, seed_document):
        doc_id = seed_document(user_id="user-1")
        r = client.get(f"{API}/document/{doc_id}", headers=user_headers)
        assert r.status_code == 200
        assert r.json()["id"] == doc_id

    def test_other_user_cannot_read_document(self, client, user_headers, seed_document):
        doc_id = seed_document(user_id="user-2")          # owned by someone else
        r = client.get(f"{API}/document/{doc_id}", headers=user_headers)
        assert r.status_code == 404                        # 404, not 403 (no existence leak)

    def test_other_user_cannot_chat_document(self, client, user_headers, seed_document):
        doc_id = seed_document(user_id="user-2")
        r = client.post(
            f"{API}/chat",
            headers=user_headers,
            json={"document_id": doc_id, "message": "hello"},
        )
        assert r.status_code == 404

    def test_other_user_cannot_delete_document(self, client, user_headers, seed_document):
        doc_id = seed_document(user_id="user-2")
        r = client.delete(f"{API}/document/{doc_id}", headers=user_headers)
        assert r.status_code == 404

    def test_list_only_returns_own_documents(self, client, user_headers, seed_document):
        mine = seed_document(user_id="user-1")
        theirs = seed_document(user_id="user-2")
        ids = [d["id"] for d in client.get(f"{API}/documents", headers=user_headers).json()]
        assert mine in ids
        assert theirs not in ids


# ── Admin authorization ───────────────────────────────────────────────────────

class TestAdminAuthorization:
    def test_admin_endpoint_requires_auth(self, client):
        assert client.get(f"{API}/admin/stats").status_code == 401

    def test_non_admin_forbidden(self, client, user_headers):
        assert client.get(f"{API}/admin/stats", headers=user_headers).status_code == 403

    def test_forged_admin_claim_rejected(self, client):
        # admin role, but signed with the wrong secret → signature check fails.
        forged = make_token(user_id="evil", role="admin", secret="attacker-secret")
        r = client.get(f"{API}/admin/stats", headers={"Authorization": f"Bearer {forged}"})
        assert r.status_code == 401

    def test_real_admin_allowed(self, client, admin_headers):
        assert client.get(f"{API}/admin/stats", headers=admin_headers).status_code == 200


# ── Upload guards ─────────────────────────────────────────────────────────────

class TestUploadGuards:
    def test_oversized_upload_rejected(self, client, user_headers, monkeypatch):
        # Force a 0 MB ceiling so any non-empty body is rejected fast.
        monkeypatch.setattr(cfg.get_settings(), "max_file_size_mb", 0)
        r = client.post(
            f"{API}/upload",
            headers=user_headers,
            files={"file": ("doc.pdf", _pdf_bytes(), "application/pdf")},
        )
        assert r.status_code == 413

    def test_content_extension_mismatch_rejected(self, client, user_headers, monkeypatch, tmp_path):
        # Write uploads to a temp dir so the test doesn't litter the repo.
        monkeypatch.setattr(cfg.get_settings(), "upload_dir", tmp_path)
        # A file claiming to be a PDF but whose bytes are not a PDF.
        r = client.post(
            f"{API}/upload",
            headers=user_headers,
            files={"file": ("fake.pdf", b"this is definitely not a pdf", "application/pdf")},
        )
        # Validation runs inside save_upload and marks the doc as errored.
        assert r.status_code == 201
        body = r.json()
        assert body["status"] == "error"
        assert "extension" in (body.get("message") or "").lower()

    def test_unsupported_extension_rejected(self, client, user_headers):
        r = client.post(
            f"{API}/upload",
            headers=user_headers,
            files={"file": ("x.exe", b"MZ\x90\x00", "application/octet-stream")},
        )
        assert r.status_code == 400


# ── Prompt / question length cap ──────────────────────────────────────────────

class TestQuestionLengthCap:
    def test_route_level_question_cap(self, client, user_headers, seed_document, monkeypatch):
        doc_id = seed_document(user_id="user-1")
        monkeypatch.setattr(cfg.get_settings(), "max_question_length", 5)
        r = client.post(
            f"{API}/chat",
            headers=user_headers,
            json={"document_id": doc_id, "message": "this is longer than five"},
        )
        assert r.status_code == 400

    def test_schema_level_message_cap(self, client, user_headers, seed_document):
        doc_id = seed_document(user_id="user-1")
        r = client.post(
            f"{API}/chat",
            headers=user_headers,
            json={"document_id": doc_id, "message": "x" * 5000},  # > schema max_length
        )
        assert r.status_code == 422
