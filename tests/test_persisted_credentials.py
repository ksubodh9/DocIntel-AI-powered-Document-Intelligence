"""
Tests for Step 4 BYOK (saved/persisted keys): encryption, the CredentialService,
the management endpoints, and saved-key resolution. Provider calls are mocked.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet

from app.config.settings import get_settings
from app.utils import crypto
from app.services.credential_service import CredentialService, UnsupportedProviderError
from app.api.routes import _resolve_llm_service
from tests.conftest import TestSessionLocal


@pytest.fixture
def enc_key():
    """Configure a real Fernet key on the settings singleton for the test."""
    key = Fernet.generate_key().decode()
    s = get_settings()
    old = s.credentials_encryption_key
    object.__setattr__(s, "credentials_encryption_key", key)
    crypto.get_cipher.cache_clear()
    yield key
    object.__setattr__(s, "credentials_encryption_key", old)
    crypto.get_cipher.cache_clear()


@pytest.fixture
def no_enc_key():
    s = get_settings()
    old = s.credentials_encryption_key
    object.__setattr__(s, "credentials_encryption_key", "")
    crypto.get_cipher.cache_clear()
    yield
    object.__setattr__(s, "credentials_encryption_key", old)
    crypto.get_cipher.cache_clear()


def _db():
    return TestSessionLocal()


class TestCrypto:
    def test_round_trip(self, enc_key):
        c = crypto.get_cipher()
        token = c.encrypt("sk-secret-123")
        assert token != "sk-secret-123"
        assert c.decrypt(token) == "sk-secret-123"

    def test_available_flag(self, enc_key):
        assert crypto.encryption_available() is True

    def test_unavailable_without_key(self, no_enc_key):
        assert crypto.encryption_available() is False
        with pytest.raises(crypto.CredentialsEncryptionUnavailable):
            crypto.get_cipher()


class TestCredentialService:
    def test_save_encrypts_and_decrypts(self, enc_key):
        db = _db()
        try:
            svc = CredentialService(db)
            cred = svc.save("u-svc-1", "openai", "sk-plain-9999", model="gpt-4o")
            assert cred.ciphertext != "sk-plain-9999"   # stored encrypted
            assert cred.key_last4 == "9999"
            assert svc.decrypt_key(cred) == "sk-plain-9999"
        finally:
            db.close()

    def test_only_one_active(self, enc_key):
        db = _db()
        try:
            svc = CredentialService(db)
            svc.save("u-svc-2", "openai", "sk-aaaa")
            svc.save("u-svc-2", "gemini", "AIzaBBBB")
            actives = [c for c in svc.list_for_user("u-svc-2") if c.is_active]
            assert len(actives) == 1
            assert actives[0].provider == "gemini"      # latest save wins
        finally:
            db.close()

    def test_delete(self, enc_key):
        db = _db()
        try:
            svc = CredentialService(db)
            svc.save("u-svc-3", "openai", "sk-cccc")
            assert svc.delete("u-svc-3", "openai") is True
            assert svc.delete("u-svc-3", "openai") is False
        finally:
            db.close()

    def test_unsupported_provider(self, enc_key):
        db = _db()
        try:
            with pytest.raises(UnsupportedProviderError):
                CredentialService(db).save("u-svc-4", "bogus", "x")
        finally:
            db.close()

    def test_save_refused_without_encryption(self, no_enc_key):
        db = _db()
        try:
            with pytest.raises(crypto.CredentialsEncryptionUnavailable):
                CredentialService(db).save("u-svc-5", "openai", "sk-x")
        finally:
            db.close()


class TestEndpoints:
    def test_session_mode_requires_auth(self, client):
        assert client.get("/api/v1/session/mode").status_code == 401

    def test_session_mode_shape(self, client, user_headers, enc_key):
        r = client.get("/api/v1/session/mode", headers=user_headers)
        assert r.status_code == 200
        body = r.json()
        assert body["persistence_available"] is True
        assert "openai" in body["supported_providers"]

    def test_save_list_delete_flow(self, client, user_headers, enc_key):
        with patch("app.api.routes.LLMService.complete", return_value="pong"):
            r = client.post("/api/v1/credentials", headers=user_headers,
                            json={"provider": "openai", "api_key": "sk-secret-1234"})
        assert r.status_code == 201
        item = r.json()
        assert "1234" in item["masked_key"]
        assert "sk-secret-1234" not in item["masked_key"]
        assert item["is_active"] is True

        r = client.get("/api/v1/credentials", headers=user_headers)
        assert r.status_code == 200
        body = r.json()
        assert body["encryption_available"] is True
        assert any(c["provider"] == "openai" for c in body["credentials"])
        # real key never returned anywhere in the listing
        assert "sk-secret-1234" not in r.text

        r = client.delete("/api/v1/credentials/openai", headers=user_headers)
        assert r.status_code == 204
        r = client.delete("/api/v1/credentials/openai", headers=user_headers)
        assert r.status_code == 404

    def test_save_blocked_without_encryption(self, client, user_headers, no_enc_key):
        r = client.post("/api/v1/credentials", headers=user_headers,
                        json={"provider": "openai", "api_key": "sk-x", "validate_first": False})
        assert r.status_code == 503

    def test_validation_failure_is_not_saved(self, client, user_headers, enc_key):
        from app.services.llm_service import LLMError
        with patch("app.api.routes.LLMService.complete", side_effect=LLMError("Auth error. sk-bad")):
            r = client.post("/api/v1/credentials", headers=user_headers,
                            json={"provider": "openai", "api_key": "sk-bad", "validate_first": True})
        assert r.status_code == 400
        assert "sk-bad" not in r.text
        # nothing persisted
        r = client.get("/api/v1/credentials", headers=user_headers)
        assert all(c["provider"] != "openai" for c in r.json()["credentials"])


class TestSavedKeyResolution:
    def _req(self, headers=None):
        return SimpleNamespace(headers=headers or {})

    def test_resolver_uses_active_saved_key(self, enc_key):
        db = _db()
        try:
            CredentialService(db).save("u-res-1", "anthropic", "sk-saved-7777", model="claude-3-haiku-20240307")
            svc = _resolve_llm_service(self._req(), db=db, user={"user_id": "u-res-1"})
            assert svc.provider == "anthropic"
            assert svc.cfg.key_for("anthropic") == "sk-saved-7777"
        finally:
            db.close()

    def test_use_default_header_bypasses_saved_key(self, enc_key):
        db = _db()
        try:
            CredentialService(db).save("u-res-2", "anthropic", "sk-saved-8888")
            svc = _resolve_llm_service(
                self._req({"X-LLM-Use-Default": "true"}),
                db=db, user={"user_id": "u-res-2"},
            )
            assert svc.provider == get_settings().llm_provider   # server default, not saved
        finally:
            db.close()

    def test_session_header_takes_precedence_over_saved(self, enc_key):
        db = _db()
        try:
            CredentialService(db).save("u-res-3", "anthropic", "sk-saved-9999")
            svc = _resolve_llm_service(
                self._req({"X-LLM-Provider": "openai", "X-LLM-Api-Key": "sk-session"}),
                db=db, user={"user_id": "u-res-3"},
            )
            assert svc.provider == "openai"
            assert svc.cfg.key_for("openai") == "sk-session"
        finally:
            db.close()
