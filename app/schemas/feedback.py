"""
Pydantic schemas for user feedback.

The submission payload is deliberately tiny — rating and/or comment — so the
form stays a two-field, low-effort ask. `category` defaults server-side.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator

# Keep the comment bound aligned with the textarea maxLength on the client.
MAX_COMMENT_LENGTH = 2000


class FeedbackCreate(BaseModel):
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    comment: Optional[str] = Field(default=None, max_length=MAX_COMMENT_LENGTH)

    # Retained but optional; the UI doesn't collect it today.
    category: str = "general"

    # Silently-attached context.
    route: Optional[str] = Field(default=None, max_length=255)
    last_feature_used: Optional[str] = Field(default=None, max_length=50)

    @model_validator(mode="after")
    def _require_something(self):
        has_comment = bool(self.comment and self.comment.strip())
        if self.rating is None and not has_comment:
            raise ValueError("Provide a rating or a comment.")
        return self


class FeedbackResponse(BaseModel):
    id: str
    status: str = "new"
    message: str = "Thanks for the feedback!"


class AdminFeedbackItem(BaseModel):
    id: str
    user_id: Optional[str] = None
    email: Optional[str] = None
    category: str
    rating: Optional[int] = None
    comment: Optional[str] = None
    route: Optional[str] = None
    last_feature_used: Optional[str] = None
    status: str
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}
