"""
Symmetric encryption for saved BYOK API keys.

Keys are encrypted at rest with Fernet (AES-128-CBC + HMAC). The Fernet key
comes from CREDENTIALS_ENCRYPTION_KEY and is never stored in the database. If
the setting is empty, persistence is considered disabled — saving keys must be
refused (session-only BYOK still works).
"""

from functools import lru_cache

from app.config.settings import get_settings


class CredentialsEncryptionUnavailable(Exception):
    """Raised when an encryption operation is attempted but no key is configured."""


class CredentialCipher:
    """Thin wrapper over Fernet. Encrypts/decrypts UTF-8 strings."""

    def __init__(self, fernet_key: str):
        from cryptography.fernet import Fernet

        # Validates the key format; raises ValueError on a malformed key.
        self._fernet = Fernet(fernet_key.encode() if isinstance(fernet_key, str) else fernet_key)

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, token: str) -> str:
        return self._fernet.decrypt(token.encode()).decode()


@lru_cache(maxsize=1)
def get_cipher() -> CredentialCipher:
    """
    Return the configured cipher singleton.
    Raises CredentialsEncryptionUnavailable if CREDENTIALS_ENCRYPTION_KEY is unset.
    """
    key = get_settings().credentials_encryption_key
    if not key:
        raise CredentialsEncryptionUnavailable(
            "CREDENTIALS_ENCRYPTION_KEY is not set — saving API keys is disabled."
        )
    return CredentialCipher(key)


def encryption_available() -> bool:
    """True if a (well-formed) encryption key is configured."""
    try:
        get_cipher()
        return True
    except Exception:
        return False
