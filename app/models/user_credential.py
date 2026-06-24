"""
UserCredential — a user's saved (persisted) BYOK provider key.

Only the ciphertext is stored; the plaintext key never touches the database.
`key_last4` is kept in the clear purely so the UI can show a masked hint
(e.g. "openai ••••4f2a") without having to decrypt. One active credential per
user is used by the request resolver when no per-request key header is present.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Boolean, UniqueConstraint

from app.database.base import Base


class UserCredential(Base):
    __tablename__ = "user_credentials"
    __table_args__ = (UniqueConstraint("user_id", "provider", name="uq_user_provider"),)

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)   # Supabase user UUID
    provider = Column(String(50), nullable=False)              # openai | gemini | ...
    model = Column(String(120), nullable=True)                 # optional model override
    ciphertext = Column(String, nullable=False)               # Fernet-encrypted API key
    key_last4 = Column(String(8), nullable=True)              # masked display hint only
    is_active = Column(Boolean, default=True, nullable=False)  # the one the resolver uses
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<UserCredential user={self.user_id} provider={self.provider} active={self.is_active}>"
