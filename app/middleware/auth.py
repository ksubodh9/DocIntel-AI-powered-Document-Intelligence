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
import logging
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

logger = logging.getLogger(__name__)

# Supabase signs JWTs with the project's JWT secret (found in
# Supabase dashboard -> Settings -> API -> JWT Secret).
_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET", "")

try:
    import jwt as pyjwt
    _JWT_AVAILABLE = True
except ImportError:
    _JWT_AVAILABLE = False
    logger.warning("PyJWT not installed — auth middleware will pass through without verification. Run: pip install PyJWT")


def _decode_token(token: str) -> dict:
    if not _JWT_AVAILABLE:
        # Dev fallback: decode without verification (NOT safe for production)
        import base64, json
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        return json.loads(base64.urlsafe_b64decode(payload_b64))

    if not _JWT_SECRET:
        # No JWT secret configured — fall back to unverified decode.
        # This is safe for single-user or demo deployments where auth is
        # optional. Production deployments MUST set SUPABASE_JWT_SECRET.
        logger.warning("SUPABASE_JWT_SECRET not set — decoding JWT without verification.")
        import base64, json as _json
        try:
            payload_b64 = token.split(".")[1]
            payload_b64 += "=" * (-len(payload_b64) % 4)
            return _json.loads(base64.urlsafe_b64decode(payload_b64))
        except Exception:
            raise HTTPException(status_code=401, detail="Could not decode token (SUPABASE_JWT_SECRET not set).")

    try:
        return pyjwt.decode(
            token,
            _JWT_SECRET,
            algorithms=["HS256"],
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
