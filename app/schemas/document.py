"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Citation ─────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    page: int
    text: str
    chunk_id: str


# ── Upload ────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    file_size: int
    page_count: int
    status: str
    message: str


# ── Classification ────────────────────────────────────────────────────────────

class ClassifyResponse(BaseModel):
    document_id: str
    document_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)


# ── Summary ───────────────────────────────────────────────────────────────────

class SummaryResponse(BaseModel):
    document_id: str
    short_summary: str
    detailed_summary: str
    topics: list[str]
    keywords: list[str]
    entities: list[str]


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    document_id: str
    message: str = Field(..., min_length=1, max_length=2000)
    include_history: bool = True


class ChatResponse(BaseModel):
    document_id: str
    message_id: str
    answer: str
    citations: list[Citation]
    sources_found: bool


# ── Suggested Questions ───────────────────────────────────────────────────────

class SuggestedQuestionsResponse(BaseModel):
    document_id: str
    questions: list[str]


# ── Document Detail ───────────────────────────────────────────────────────────

class DocumentDetail(BaseModel):
    id: str
    filename: str
    file_size: int
    page_count: int
    document_type: Optional[str]
    classification_confidence: Optional[float]
    status: str
    short_summary: Optional[str]
    detailed_summary: Optional[str]
    topics: Optional[list[str]]
    keywords: Optional[list[str]]
    entities: Optional[list[str]]
    suggested_questions: Optional[list[str]]
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


# ── Chat History ──────────────────────────────────────────────────────────────

class ChatHistoryItem(BaseModel):
    id: str
    role: str
    content: str
    citations: Optional[list[Citation]]
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


class ChatHistoryResponse(BaseModel):
    document_id: str
    messages: list[ChatHistoryItem]


# ── Error ─────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
