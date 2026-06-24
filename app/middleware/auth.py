"""
Supabase JWT authentication middleware for FastAPI.

Every protected endpoint depends on `get_current_user`, which:
1. Reads the Bearer token from the Authorization header
2. Verifies its signature, then returns a dict with user_id, email, and is_admin

Verification strategy (decided per-token by the JWT header `alg`):
- Asymmetric (ES256 / RS256): keys are fetched from Supabase's JWKS endpoint
  ({SUPABASE_URL}/auth/v1/.well-known/jwks.json) and cached. This is the
  default for newer Supabase projects using signing keys.
- Symmetric (HS256): verified with the legacy SUPABASE_JWT_SECRET, if set.

Admin check uses app_metadata.role = "admin".
To grant admin:
  UPDATE auth.users SET raw_app_meta_data = '{"role":"admin"}'
  WHERE email = 'admin@example.com';
"""

import os
import base64
import json
import logging
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from app.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Legacy symmetric secret (Supabase dashboard -> Settings -> API -> JWT Secret).
# Optional now — asymmetric (JWKS) verification is preferred when SUPABASE_URL is set.
_JWT_SECRET = settings.supabase_jwt_secret or os.environ.get("SUPABASE_JWT_SECRET", "")
_SUPABASE_URL = (settings.supabase_url or os.environ.get("SUPABASE_URL", "")).rstrip("/")

# Algorithms we accept. alg=none and anything outside this set is rejected.
_ASYMMETRIC_ALGS = ("ES256", "RS256")

try:
    import jwt as pyjwt
    from jwt import PyJWKClient
    _JWT_AVAILABLE = True
except ImportError:
    _JWT_AVAILABLE = False
    PyJWKClient = None  # type: ignore
    logger.error("PyJWT not installed — JWT signatures cannot be verified. Run: pip install 'PyJWT[crypto]'")


# Lazily-built, process-wide JWKS client. PyJWKClient caches fetched keys
# internally (lifespan default) so we don't hit Supabase on every request.
_jwks_client = None


def _get_jwks_client():
    global _jwks_client
    if _jwks_client is None:
        if not _SUPABASE_URL:
            return None
        jwks_url = f"{_SUPABASE_URL}/auth/v1/.well-known/jwks.json"
        _jwks_client = PyJWKClient(jwks_url)
        logger.info("Initialized Supabase JWKS client: %s", jwks_url)
    return _jwks_client


def _unverified_decode(token: str) -> dict:
    """Decode a JWT payload WITHOUT verifying the signature. Insecure — only
    reachable when allow_insecure_auth is explicitly enabled for local dev."""
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        return json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        raise HTTPException(status_code=401, detail="Could not decode token.")


def _token_alg(token: str) -> str:
    """Read the `alg` field from the JWT header without verifying anything."""
    try:
        header = pyjwt.get_unverified_header(token)
        return header.get("alg", "")
    except Exception:
        raise HTTPException(status_code=401, detail="Could not read token header.")


def _decode_asymmetric(token: str, alg: str) -> dict:
    client = _get_jwks_client()
    if client is None:
        logger.error(
            "Token uses %s but SUPABASE_URL is not set — cannot fetch JWKS to verify it.", alg
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication is not configured on the server.",
        )
    try:
        signing_key = client.get_signing_key_from_jwt(token)
    except Exception as e:
        logger.warning("Failed to resolve JWKS signing key: %s", e)
        raise HTTPException(status_code=401, detail="Could not verify token signing key.")
    return pyjwt.decode(
        token,
        signing_key.key,
        algorithms=list(_ASYMMETRIC_ALGS),
        options={"verify_aud": False},   # Supabase tokens don't always carry aud
    )


def _decode_symmetric(token: str) -> dict:
    if not _JWT_SECRET:
        logger.error(
            "Token uses HS256 but SUPABASE_JWT_SECRET is not set — cannot verify it. "
            "Either set the secret, or have the client send an asymmetric (ES256) token."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication is not configured on the server.",
        )
    return pyjwt.decode(
        token,
        _JWT_SECRET,
        algorithms=["HS256"],   # pinned — rejects alg=none and asymmetric confusion
        options={"verify_aud": False},
    )


def _decode_token(token: str) -> dict:
    # Fail closed: without PyJWT we cannot verify any signature. Permit unverified
    # decoding only when an operator explicitly opted in via ALLOW_INSECURE_AUTH.
    if not _JWT_AVAILABLE:
        if settings.allow_insecure_auth:
            logger.warning(
                "ALLOW_INSECURE_AUTH is on — decoding JWT WITHOUT signature "
                "verification. Never use this in production."
            )
            return _unverified_decode(token)
        logger.error("Refusing to accept token: PyJWT not installed and ALLOW_INSECURE_AUTH is disabled.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication is not configured on the server.",
        )

    alg = _token_alg(token)
    try:
        if alg in _ASYMMETRIC_ALGS:
            return _decode_asymmetric(token, alg)
        if alg == "HS256":
            return _decode_symmetric(token)
        raise HTTPException(status_code=401, detail=f"Unsupported token algorithm: {alg or 'none'}.")
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired. Please sign in again.")
    except pyjwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


def get_current_user(authorization: Optional[str] = Header(default=None)) -> dict:
    """
    FastAPI dependency — extracts and verifies the Supabase JWT.
    Returns { user_id, email, is_admin }.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization[len("Bearer "):]
    payload = _decode_token(token)

    user_id: str = payload.get("sub", "")
    email: str = payload.get("email", "")
    app_meta: dict = payload.get("app_metadata", {}) or {}
    is_admin: bool = app_meta.get("role") == "admin"

    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing user ID (sub).")

    return {"user_id": user_id, "email": email, "is_admin": is_admin}


def get_optional_user(authorization: Optional[str] = Header(default=None)) -> Optional[dict]:
    """
    Like get_current_user but returns None instead of raising 401.
    Use for endpoints that work both authenticated and unauthenticated.
    """
    try:
        return get_current_user(authorization)
    except HTTPException:
        return None


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """FastAPI dependency — raises 403 if the user is not an admin."""
    if not user.get("is_admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required.")
    return user
