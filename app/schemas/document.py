"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime

from app.config.settings import get_settings

# Single source of truth for the max chat-question length. Keeping the schema
# bound (422) aligned with the route-level guard (400) avoids dead code and a
# confusing mismatch between the two limits.
_MAX_QUESTION_LENGTH = get_settings().max_question_length


# ── Citation ──────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    page: int
    text: str
    chunk_id: str
    relevance_score: float = 0.0
    document_id: Optional[str] = None      # used in multi-doc chat
    document_name: Optional[str] = None    # human-readable filename


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


# ── Metadata ──────────────────────────────────────────────────────────────────

class MetadataResponse(BaseModel):
    document_id: str
    metadata: dict[str, Any]


# ── Tables ────────────────────────────────────────────────────────────────────

class TableItem(BaseModel):
    page: int
    caption: str
    markdown: str


class TablesResponse(BaseModel):
    document_id: str
    table_count: int
    tables: list[TableItem]


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    document_id: str
    message: str = Field(..., min_length=1, max_length=_MAX_QUESTION_LENGTH)
    include_history: bool = True


class MultiChatRequest(BaseModel):
    document_ids: list[str] = Field(..., min_length=1)
    message: str = Field(..., min_length=1, max_length=_MAX_QUESTION_LENGTH)
    include_history: bool = False


class ChatResponse(BaseModel):
    document_id: str
    message_id: str
    answer: str
    citations: list[Citation]
    sources_found: bool


class MultiChatResponse(BaseModel):
    document_ids: list[str]
    message_id: str
    answer: str
    citations: list[Citation]
    sources_found: bool


# ── Comparison ────────────────────────────────────────────────────────────────

class CompareRequest(BaseModel):
    document_id_a: str
    document_id_b: str


class CompareResponse(BaseModel):
    document_id_a: str
    document_id_b: str
    similarities: list[str]
    differences: list[str]
    recommendation: str
    detailed_comparison: str


# ── Suggested Questions ───────────────────────────────────────────────────────

class SuggestedQuestionsResponse(BaseModel):
    document_id: str
    questions: list[str]


# ── Document Detail ───────────────────────────────────────────────────────────

class DocumentDetail(BaseModel):
    id: str
    filename: str
    original_filename: Optional[str] = None
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
    doc_metadata: Optional[dict[str, Any]]
    has_tables: Optional[bool]
    table_count: Optional[int]
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
