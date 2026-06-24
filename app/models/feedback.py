"""
Feedback — general product feedback captured from users.

Prompted after a user has engaged with a feature, when they log out or leave
the page. The user-facing form is intentionally minimal (rating + comment) so
people don't have to spend time on it; `category` is retained at the model
level (defaulting to "general") for future routing/triage.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, String, Integer, Text, DateTime

from app.database.base import Base


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=True, index=True)   # Supabase user UUID (nullable: may be mid-logout / anon)
    email = Column(String(255), nullable=True)

    # Retained for future triage even though the form doesn't ask for it.
    category = Column(String(20), nullable=False, default="general")  # general | bug | idea

    rating = Column(Integer, nullable=True)   # 1–5, optional
    comment = Column(Text, nullable=True)

    # Lightweight context captured silently.
    route = Column(String(255), nullable=True)            # where the user was
    last_feature_used = Column(String(50), nullable=True)  # e.g. chat | summarize | compare
    user_agent = Column(String(512), nullable=True)

    # Triage state for the admin view.
    status = Column(String(20), nullable=False, default="new")  # new | reviewed

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:
        return f"<Feedback user={self.user_id} rating={self.rating} status={self.status}>"
