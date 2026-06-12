"""
Supabase JWT authentication middleware for FastAPI.

Every protected endpoint depends on `get_current_user`, which:
1. Reads the Bearer token from the Authorization header
2. Verifies it with the Supabase JWT secret (HS256)
3. Returns a dict with user_id, email, and is_admin

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

# Supabase signs JWTs with the project's JWT secret (found in
# Supabase dashboard -> Settings -> API -> JWT Secret). Prefer the typed setting,
# fall back to the raw env var for backward compatibility.
_JWT_SECRET = settings.supabase_jwt_secret or os.environ.get("SUPABASE_JWT_SECRET", "")

try:
    import jwt as pyjwt
    _JWT_AVAILABLE = True
except ImportError:
    _JWT_AVAILABLE = False
    logger.error("PyJWT not installed — JWT signatures cannot be verified. Run: pip install PyJWT")


def _unverified_decode(token: str) -> dict:
    """Decode a JWT payload WITHOUT verifying the signature. Insecure — only
    reachable when allow_insecure_auth is explicitly enabled for local dev."""
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        return json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        raise HTTPException(status_code=401, detail="Could not decode token.")


def _decode_token(token: str) -> dict:
    # Fail closed: without PyJWT or a configured secret, we cannot verify the
    # signature. We only permit unverified decoding when an operator has
    # explicitly opted in via ALLOW_INSECURE_AUTH (intended for local dev).
    if not _JWT_AVAILABLE or not _JWT_SECRET:
        if settings.allow_insecure_auth:
            logger.warning(
                "ALLOW_INSECURE_AUTH is on — decoding JWT WITHOUT signature "
                "verification. Never use this in production."
            )
            return _unverified_decode(token)
        logger.error(
            "Refusing to accept token: SUPABASE_JWT_SECRET not set or PyJWT "
            "missing, and ALLOW_INSECURE_AUTH is disabled."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication is not configured on the server.",
        )

    try:
        return pyjwt.decode(
            token,
            _JWT_SECRET,
            algorithms=["HS256"],   # pinned — rejects alg=none and asymmetric confusion
            options={"verify_aud": False},   # Supabase tokens don't always have aud
        )
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
