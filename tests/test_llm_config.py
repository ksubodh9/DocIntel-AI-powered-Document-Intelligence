"""
Unit tests for the BYOK foundation: LLMConfig resolution, provider allowlist,
and secret scrubbing. No network — we never actually call a provider.
"""

import pytest

from app.services.llm_service import (
    LLMConfig,
    LLMService,
    LLMError,
    ALLOWED_PROVIDERS,
    _scrub,
    get_llm_service,
)
from app.config.settings import get_settings


class TestFromSettings:
    def test_maps_keys_models_and_fallback(self):
        cfg = LLMConfig.from_settings(get_settings())
        s = get_settings()
        assert cfg.provider == s.llm_provider
        # Every known provider has a key slot (possibly empty) and a default model.
        assert set(cfg.api_keys) >= {"openai", "gemini", "anthropic", "groq", "huggingface"}
        assert cfg.models["gemini"] == s.gemini_model

    def test_models_for_prefers_model_list(self):
        cfg = LLMConfig(
            provider="groq",
            models={"groq": "single"},
            model_lists={"groq": ["a", "b"]},
        )
        assert cfg.models_for("groq") == ["a", "b"]

    def test_models_for_falls_back_to_single_model(self):
        cfg = LLMConfig(provider="openai", models={"openai": "gpt-4o-mini"})
        assert cfg.models_for("openai") == ["gpt-4o-mini"]


class TestForByok:
    def test_builds_single_provider_config(self):
        cfg = LLMConfig.for_byok("openai", "sk-test", model="gpt-4o")
        assert cfg.provider == "openai"
        assert cfg.fallback_chain == []          # no cross-provider fallback
        assert cfg.key_for("openai") == "sk-test"
        assert cfg.models_for("openai") == ["gpt-4o"]

    def test_falls_back_to_server_default_model(self):
        cfg = LLMConfig.for_byok("gemini", "AIza-test")
        # No model supplied -> uses the server's default for that provider.
        assert cfg.models_for("gemini") == [get_settings().gemini_model]

    def test_rejects_unknown_provider(self):
        with pytest.raises(LLMError):
            LLMConfig.for_byok("totally-not-a-provider", "key")

    def test_normalizes_provider_case(self):
        cfg = LLMConfig.for_byok("OpenAI", "sk-test")
        assert cfg.provider == "openai"


class TestServiceUsesInjectedConfig:
    def test_build_chain_uses_config(self):
        cfg = LLMConfig(provider="groq", fallback_chain=["ollama"])
        svc = LLMService(cfg)
        assert svc.provider == "groq"
        assert svc._build_chain() == ["groq", "ollama"]

    def test_byok_service_has_no_fallback(self):
        svc = LLMService(LLMConfig.for_byok("anthropic", "sk-x"))
        assert svc._build_chain() == ["anthropic"]

    def test_default_service_uses_server_settings(self):
        svc = get_llm_service()
        assert svc.provider == get_settings().llm_provider

    def test_unknown_provider_call_is_rejected(self):
        svc = LLMService(LLMConfig(provider="bogus"))
        with pytest.raises(LLMError):
            svc._call_provider("bogus", "p", None, 0.1, 100)


class TestScrub:
    @pytest.mark.parametrize("secret", [
        "sk-abcdef1234567890",
        "AIzaSyABCDEF1234567890",
        "gsk_abcdef1234567890",
        "hf_abcdef1234567890",
    ])
    def test_redacts_keys(self, secret):
        out = _scrub(f"error with key {secret} in request")
        assert secret not in out
        assert "REDACTED" in out

    def test_passes_through_clean_text(self):
        assert _scrub("rate limit exceeded (429)") == "rate limit exceeded (429)"


def test_allowlist_contents():
    assert ALLOWED_PROVIDERS == {
        "gemini", "ollama", "groq", "huggingface", "openai", "anthropic"
    }
