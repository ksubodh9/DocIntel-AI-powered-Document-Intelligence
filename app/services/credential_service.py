"""
CredentialService — CRUD for persisted (saved) BYOK provider keys.

The plaintext key is encrypted before it touches the database and decrypted only
in memory when a request needs it. Saving requires CREDENTIALS_ENCRYPTION_KEY to
be configured; without it, persistence is refused and only session-only BYOK is
available.

Exactly one credential per user is marked active. The request resolver uses the
active credential when a request carries no per-request key header.
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.models.user_credential import UserCredential
from app.services.llm_service import ALLOWED_PROVIDERS
from app.utils.crypto import get_cipher, encryption_available

logger = logging.getLogger(__name__)


class UnsupportedProviderError(ValueError):
    """Raised when a provider outside the allowlist is supplied."""


class CredentialService:
    def __init__(self, db: Session):
        self.db = db

    # ---- queries ------------------------------------------------------------

    def list_for_user(self, user_id: str) -> list[UserCredential]:
        return (
            self.db.query(UserCredential)
            .filter(UserCredential.user_id == user_id)
            .order_by(UserCredential.created_at.desc())
            .all()
        )

    def get_active(self, user_id: str) -> Optional[UserCredential]:
        return (
            self.db.query(UserCredential)
            .filter(UserCredential.user_id == user_id, UserCredential.is_active.is_(True))
            .first()
        )

    def _get(self, user_id: str, provider: str) -> Optional[UserCredential]:
        return (
            self.db.query(UserCredential)
            .filter(UserCredential.user_id == user_id, UserCredential.provider == provider)
            .first()
        )

    # ---- mutations ----------------------------------------------------------

    def save(self, user_id: str, provider: str, api_key: str, model: Optional[str] = None) -> UserCredential:
        """
        Encrypt and persist a key for (user, provider). Upserts: re-saving a
        provider replaces the stored key. The saved credential becomes the
        user's single active credential.
        """
        provider = (provider or "").strip().lower()
        if provider not in ALLOWED_PROVIDERS:
            raise UnsupportedProviderError(provider)
        if not encryption_available():
            # Caller maps this to a clear 503 — never silently store plaintext.
            from app.utils.crypto import CredentialsEncryptionUnavailable
            raise CredentialsEncryptionUnavailable()

        ciphertext = get_cipher().encrypt(api_key)
        last4 = api_key[-4:] if len(api_key) >= 4 else None

        # Deactivate all other credentials so exactly one stays active.
        self.db.query(UserCredential).filter(
            UserCredential.user_id == user_id
        ).update({UserCredential.is_active: False})

        cred = self._get(user_id, provider)
        if cred:
            cred.ciphertext = ciphertext
            cred.key_last4 = last4
            cred.model = model
            cred.is_active = True
        else:
            cred = UserCredential(
                user_id=user_id, provider=provider, ciphertext=ciphertext,
                key_last4=last4, model=model, is_active=True,
            )
            self.db.add(cred)
        self.db.commit()
        self.db.refresh(cred)
        logger.info(f"[Credentials] Saved key for user={user_id} provider={provider}")
        return cred

    def delete(self, user_id: str, provider: str) -> bool:
        """Delete a stored credential. Returns True if one was removed."""
        provider = (provider or "").strip().lower()
        cred = self._get(user_id, provider)
        if not cred:
            return False
        self.db.delete(cred)
        self.db.commit()
        logger.info(f"[Credentials] Deleted key for user={user_id} provider={provider}")
        return True

    # ---- decryption (in-memory only) ---------------------------------------

    def decrypt_key(self, cred: UserCredential) -> str:
        return get_cipher().decrypt(cred.ciphertext)

    def touch_last_used(self, cred: UserCredential) -> None:
        cred.last_used_at = datetime.utcnow()
        try:
            self.db.commit()
        except Exception:
            self.db.rollback()
