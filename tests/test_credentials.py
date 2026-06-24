"""
Tests for Step 3 BYOK (session-only): the header resolver and the validate
endpoint. No network — provider calls are mocked.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.api.routes import _resolve_llm_service
from app.services.llm_service import LLMError
from app.config.settings import get_settings
from tests.conftest import TestSessionLocal


def _request(headers: dict):
    """Minimal stand-in for a Starlette Request (resolver only reads .headers.get)."""
    return SimpleNamespace(headers=headers)


class TestResolver:
    def test_byok_headers_build_single_provider_service(self):
        svc = _resolve_llm_service(_request({
            "X-LLM-Provider": "openai",
            "X-LLM-Api-Key": "sk-test",
        }))
        assert svc.provider == "openai"
        assert svc._build_chain() == ["openai"]       # no cross-provider fallback
        assert svc.cfg.key_for("openai") == "sk-test"

    def test_model_header_overrides_default(self):
        svc = _resolve_llm_service(_request({
            "X-LLM-Provider": "openai",
            "X-LLM-Api-Key": "sk-test",
            "X-LLM-Model": "gpt-4o",
        }))
        assert svc.cfg.models_for("openai") == ["gpt-4o"]

    def test_no_headers_falls_back_to_server_default(self):
        db = TestSessionLocal()
        try:
            svc = _resolve_llm_service(_request({}), db=db, user={"user_id": "no-creds-user"})
            assert svc.provider == get_settings().llm_provider
        finally:
            db.close()

    def test_partial_headers_ignored(self):
        # Provider without key -> not BYOK, use server default (no saved creds for this user).
        db = TestSessionLocal()
        try:
            svc = _resolve_llm_service(_request({"X-LLM-Provider": "openai"}),
                                       db=db, user={"user_id": "no-creds-user"})
            assert svc.provider == get_settings().llm_provider
        finally:
            db.close()

    def test_unsupported_provider_raises_400(self):
        with pytest.raises(HTTPException) as exc:
            _resolve_llm_service(_request({
                "X-LLM-Provider": "bogus",
                "X-LLM-Api-Key": "x",
            }))
        assert exc.value.status_code == 400


class TestValidateEndpoint:
    URL = "/api/v1/credentials/validate"

    def test_requires_auth(self, client):
        r = client.post(self.URL, json={"provider": "openai", "api_key": "sk-x"})
        assert r.status_code == 401

    def test_valid_key_returns_true(self, client, user_headers):
        with patch("app.api.routes.LLMService.complete", return_value="pong"):
            r = client.post(self.URL, headers=user_headers,
                            json={"provider": "openai", "api_key": "sk-x"})
        assert r.status_code == 200
        body = r.json()
        assert body["valid"] is True
        assert body["provider"] == "openai"

    def test_bad_key_returns_false_without_leaking(self, client, user_headers):
        err = LLMError("Auth error (openai). Check OPENAI_API_KEY in .env. sk-secret123")
        with patch("app.api.routes.LLMService.complete", side_effect=err):
            r = client.post(self.URL, headers=user_headers,
                            json={"provider": "openai", "api_key": "sk-secret123"})
        assert r.status_code == 200
        body = r.json()
        assert body["valid"] is False
        # The generic user_message must not leak the key or .env detail.
        assert "sk-secret123" not in body["message"]
        assert ".env" not in body["message"]

    def test_unsupported_provider_returns_false(self, client, user_headers):
        r = client.post(self.URL, headers=user_headers,
                        json={"provider": "bogus", "api_key": "x"})
        assert r.status_code == 200
        assert r.json()["valid"] is False
