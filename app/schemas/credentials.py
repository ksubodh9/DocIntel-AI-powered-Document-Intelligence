"""
Schemas for Bring-Your-Own-Key (BYOK) credential handling.

Two modes the client can choose after login:
  * Session-only — provider + key sent per request via headers
    (X-LLM-Provider / X-LLM-Api-Key / X-LLM-Model); never persisted.
  * Saved        — provider + key POSTed to /credentials, encrypted at rest,
    reused automatically on later requests.

The validate endpoint takes credentials in the body because it is an explicit,
one-off "test these credentials" action rather than a document operation.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


# ── Validate ──────────────────────────────────────────────────────────────────

class CredentialValidateRequest(BaseModel):
    provider: str = Field(..., description="openai | gemini | anthropic | groq | huggingface | ollama")
    api_key: str = Field(..., min_length=1, description="API key for the provider")
    model: Optional[str] = Field(default=None, description="Optional model name override")


class CredentialValidateResponse(BaseModel):
    valid: bool = Field(..., description="True if a test call to the provider succeeded")
    provider: str = Field(..., description="The (normalized) provider that was tested")
    message: str = Field(..., description="Human-readable result. Never contains the key or internal detail.")


# ── Save (persisted) ──────────────────────────────────────────────────────────

class SaveCredentialRequest(BaseModel):
    provider: str = Field(..., description="openai | gemini | anthropic | groq | huggingface | ollama")
    api_key: str = Field(..., min_length=1, description="API key to encrypt and store")
    model: Optional[str] = Field(default=None, description="Optional model name override")
    validate_first: bool = Field(default=True, description="Test the key with the provider before saving")


class StoredCredentialItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    provider: str
    model: Optional[str] = None
    masked_key: str = Field(..., description="Masked hint, e.g. '••••4f2a' — never the real key")
    is_active: bool
    created_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None


class StoredCredentialsResponse(BaseModel):
    encryption_available: bool = Field(..., description="False if the server has no encryption key configured")
    credentials: list[StoredCredentialItem]


# ── Post-login session bootstrap ──────────────────────────────────────────────

class SessionModeResponse(BaseModel):
    """
    Tells the client what to offer after login:
      - continue with the server's default keys, or
      - bring your own (session-only and/or saved).
    """
    server_default_available: bool = Field(..., description="Server has its own keys, so 'Continue' is possible")
    persistence_available: bool = Field(..., description="Server can store keys (encryption configured)")
    has_saved_credentials: bool = Field(..., description="This user already has a saved key")
    active_provider: Optional[str] = Field(default=None, description="Provider of the user's active saved key, if any")
    supported_providers: list[str] = Field(..., description="Providers accepted for BYOK")
