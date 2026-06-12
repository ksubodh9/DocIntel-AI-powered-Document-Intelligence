"""
FastAPI route definitions.

Endpoints:
  POST   /upload                    Upload a PDF
  POST   /classify/{doc_id}         Classify the document
  POST   /summarize/{doc_id}        Generate summaries + topics
  POST   /metadata/{doc_id}         Extract document metadata
  GET    /tables/{doc_id}           Get extracted tables
  POST   /chat                      Chat with the document (RAG Q&A)
  POST   /chat/multi                Chat across multiple documents
  POST   /compare                   Compare two documents
  GET    /document/{doc_id}         Get full document details
  GET    /document/{doc_id}/history Get chat history
  GET    /questions/{doc_id}        Get suggested questions
  GET    /documents                 List all ready documents
  DELETE /document/{doc_id}         Delete a document
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Request
from fastapi import status as http_status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.database.base import get_db
from app.services.llm_service import get_llm_service, LLMService, LLMError
from app.services.document_service import DocumentService
from app.rag.vectorstore import delete_collection
from app.middleware.auth import get_current_user
from app.models.usage import UsageEvent
from app.schemas.document import (
    UploadResponse,
    ClassifyResponse,
    SummaryResponse,
    MetadataResponse,
    TablesResponse,
    TableItem,
    ChatRequest,
    MultiChatRequest,
    ChatResponse,
    MultiChatResponse,
    CompareRequest,
    CompareResponse,
    DocumentDetail,
    ChatHistoryResponse,
    ChatHistoryItem,
    SuggestedQuestionsResponse,
    Citation,
)
from app.config.settings import get_settings

settings = get_settings()
router = APIRouter()


def get_document_service(
    db: Session = Depends(get_db),
    llm: LLMService = Depends(get_llm_service),
) -> DocumentService:
    return DocumentService(db=db, llm=llm)


def _track(db: Session, user: Optional[dict], event_type: str, doc_id: Optional[str] = None):
    """Fire-and-forget usage event. Never raises."""
    try:
        ev = UsageEvent(
            user_id=user.get("user_id") if user else None,
            document_id=doc_id,
            event_type=event_type,
        )
        db.add(ev)
        db.commit()
    except Exception:
        pass  # analytics must never break the main request


def _get_doc_or_404(doc_id: str, svc: DocumentService, user: dict):
    """Fetch a document and enforce ownership.

    Returns 404 (not 403) when the document belongs to another user so the
    endpoint never confirms the existence of resources the caller can't access.
    Admins bypass the ownership check.
    """
    doc = svc.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    if not user.get("is_admin") and doc.user_id != user.get("user_id"):
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return doc


def _require_ready(doc):
    if doc.status == "error":
        raise HTTPException(
            status_code=422,
            detail=f"Document processing failed: {doc.error_message}",
        )
    if doc.status != "ready":
        raise HTTPException(
            status_code=409,
            detail=f"Document is still being processed (status: {doc.status}). Try again shortly.",
        )


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    svc: DocumentService = Depends(get_document_service),
    user: dict = Depends(get_current_user),
):
    """
    Upload a document (PDF, Word, PowerPoint, Excel, CSV, TXT, Markdown, HTML, or image).
    The file is validated, text is extracted, and chunks are indexed in ChromaDB.
    """
    from app.utils.doc_utils import SUPPORTED_EXTENSIONS
    from pathlib import Path as _Path

    filename = file.filename or "document"
    ext = _Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported formats: {supported}",
        )

    max_bytes = settings.max_file_size_mb * 1024 * 1024

    # Early rejection: refuse oversized uploads via Content-Length BEFORE reading
    # the body into memory, so a huge payload can't exhaust RAM.
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_file_size_mb} MB.",
        )

    # Read body in bounded chunks and abort as soon as the limit is exceeded
    # (Content-Length can be absent or spoofed, so we also enforce while reading).
    file_bytes = b""
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        file_bytes += chunk
        if len(file_bytes) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {settings.max_file_size_mb} MB.",
            )

    # save_upload does blocking work (disk write, extraction, embedding) -
    # run it in a worker thread so the event loop (and /health) stays responsive.
    user_id = user.get("user_id")
    doc = await run_in_threadpool(svc.save_upload, file_bytes, filename, user_id)

    _track(svc.db, user, "upload", doc.id)
    return UploadResponse(
        document_id=doc.id,
        filename=doc.original_filename,
        file_size=doc.file_size,
        page_count=doc.page_count,
        status=doc.status,
        message=(
            "Document uploaded and indexed successfully."
            if doc.status == "ready"
            else doc.error_message or "Processing failed."
        ),
    )


# ── Classify ──────────────────────────────────────────────────────────────────

# NOTE: the handlers below are plain `def` (not `async def`) on purpose.
# They call blocking, CPU-bound work (Ollama inference, embeddings, ChromaDB,
# PDF parsing). FastAPI runs sync handlers in a threadpool, so a slow request
# no longer freezes the event loop and /health keeps responding.

@router.post("/classify/{doc_id}", response_model=ClassifyResponse)
def classify_document(
    doc_id: str,
    svc: DocumentService = Depends(get_document_service),
    user: dict = Depends(get_current_user),
):
    doc = _get_doc_or_404(doc_id, svc, user)
    _require_ready(doc)

    try:
        doc = svc.classify_document(doc)
    except LLMError as e:
        raise HTTPException(status_code=429 if e.retry_after else 502, detail={"message": str(e), "retry_after": e.retry_after})

    _track(svc.db, user, "classify", doc_id)
    return ClassifyResponse(
        document_id=doc.id,
        document_type=doc.document_type,
        confidence=doc.classification_confidence,
    )


# ── Summarize ─────────────────────────────────────────────────────────────────

@router.post("/summarize/{doc_id}", response_model=SummaryResponse)
def summarize_document(
    doc_id: str,
    svc: DocumentService = Depends(get_document_service),
    user: dict = Depends(get_current_user),
):
    doc = _get_doc_or_404(doc_id, svc, user)
    _require_ready(doc)

    try:
        doc = svc.summarize_document(doc)
        if not doc.suggested_questions:
            doc = svc.generate_suggested_questions(doc)
    except LLMError as e:
        raise HTTPException(status_code=429 if e.retry_after else 502, detail={"message": str(e), "retry_after": e.retry_after})

    _track(svc.db, user, "summarize", doc_id)
    return SummaryResponse(
        document_id=doc.id,
        short_summary=doc.short_summary or "",
        detailed_summary=doc.detailed_summary or "",
        topics=doc.topics or [],
        keywords=doc.keywords or [],
        entities=doc.entities or [],
    )


# ── Chat ──────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
def chat_with_document(
    request: ChatRequest,
    svc: DocumentService = Depends(get_document_service),
    user: dict = Depends(get_current_user),
):
    doc = _get_doc_or_404(request.document_id, svc, user)
    _require_ready(doc)

    # Cap question length to limit prompt-injection surface and token abuse.
    if len(request.message or "") > settings.max_question_length:
        raise HTTPException(
            status_code=400,
            detail=f"Question too long. Maximum is {settings.max_question_length} characters.",
        )

    try:
        result = svc.chat(doc, request.message, request.include_history)
    except LLMError as e:
        raise HTTPException(status_code=429 if e.retry_after else 502, detail={"message": str(e), "retry_after": e.retry_after})
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"[Chat] Unexpected error: {e}", exc_info=True)
        # Don't leak internal exception detail to the client.
        raise HTTPException(status_code=500, detail={"message": "Chat failed due to an internal error."})

    _track(svc.db, user, "chat", request.document_id)
    citations = [Citation(**c) for c in result["citations"]]
    return ChatResponse(
        document_id=doc.id,
        message_id=result["message_id"],
        answer=result["answer"],
        citations=citations,
        sources_found=result["sources_found"],
    )


# ── Document Detail ───────────────────────────────────────────────────────────

@router.get("/document/{doc_id}", response_model=DocumentDetail)
def get_document(
    doc_id: str,
    svc: DocumentService = Depends(get_document_service),
    user: dict = Depends(get_current_user),
):
    doc = _get_doc_or_404(doc_id, svc, user)
    return DocumentDetail.model_validate(doc)


# ── Chat History ──────────────────────────────────────────────────────────────

@router.get("/document/{doc_id}/history", response_model=ChatHistoryResponse)
def get_chat_history(
    doc_id: str,
    svc: DocumentService = Depends(get_document_service),
    user: dict = Depends(get_current_user),
):
    _get_doc_or_404(doc_id, svc, user)
    messages = svc.get_chat_history(doc_id)
    return ChatHistoryResponse(
        document_id=doc_id,
        messages=[ChatHistoryItem.model_validate(m) for m in messages],
    )


# ── Suggested Questions ───────────────────────────────────────────────────────

@router.get("/questions/{doc_id}", response_model=SuggestedQuestionsResponse)
def get_suggested_questions(
    doc_id: str,
    svc: DocumentService = Depends(get_document_service),
    user: dict = Depends(get_current_user),
):
    doc = _get_doc_or_404(doc_id, svc, user)
    _require_ready(doc)

    if not doc.suggested_questions:
        try:
            doc = svc.generate_suggested_questions(doc)
        except LLMError as e:
            raise HTTPException(status_code=502, detail=str(e))

    return SuggestedQuestionsResponse(
        document_id=doc.id,
        questions=doc.suggested_questions or [],
    )


# ── Metadata ──────────────────────────────────────────────────────────────────

@router.post("/metadata/{doc_id}", response_model=MetadataResponse)
def extract_metadata(
    doc_id: str,
    svc: DocumentService = Depends(get_document_service),
    user: dict = Depends(get_current_user),
):
    doc = _get_doc_or_404(doc_id, svc, user)
    _require_ready(doc)
    try:
        doc = svc.extract_metadata(doc)
    except LLMError as e:
        raise HTTPException(status_code=429 if e.retry_after else 502, detail={"message": str(e), "retry_after": e.retry_after})
    return MetadataResponse(document_id=doc.id, metadata=doc.doc_metadata or {})


# ── Tables ────────────────────────────────────────────────────────────────────

@router.get("/tables/{doc_id}", response_model=TablesResponse)
def get_tables(
    doc_id: str,
    svc: DocumentService = Depends(get_document_service),
    user: dict = Depends(get_current_user),
):
    doc = _get_doc_or_404(doc_id, svc, user)
    _require_ready(doc)
    raw_tables = doc.tables or []
    tables = [TableItem(page=t["page"], caption=t["caption"], markdown=t["markdown"]) for t in raw_tables]
    return TablesResponse(document_id=doc.id, table_count=len(tables), tables=tables)


# ── Multi-document Chat ───────────────────────────────────────────────────────

@router.post("/chat/multi", response_model=MultiChatResponse)
def multi_chat(
    request: MultiChatRequest,
    svc: DocumentService = Depends(get_document_service),
    user: dict = Depends(get_current_user),
):
    if len(request.message or "") > settings.max_question_length:
        raise HTTPException(
            status_code=400,
            detail=f"Question too long. Maximum is {settings.max_question_length} characters.",
        )

    # Validate all documents exist, are owned by the caller, and are ready
    doc_map = {}
    for doc_id in request.document_ids:
        doc = _get_doc_or_404(doc_id, svc, user)
        if doc.status != "ready":
            raise HTTPException(status_code=409, detail=f"Document '{doc_id}' is not ready (status: {doc.status}).")
        doc_map[doc_id] = doc

    try:
        result = svc.chat_multi(request.document_ids, doc_map, request.message)
    except LLMError as e:
        raise HTTPException(status_code=429 if e.retry_after else 502, detail={"message": str(e), "retry_after": e.retry_after})

    citations = [Citation(**c) for c in result["citations"]]
    return MultiChatResponse(
        document_ids=request.document_ids,
        message_id=result["message_id"],
        answer=result["answer"],
        citations=citations,
        sources_found=result["sources_found"],
    )


# ── Document Comparison ───────────────────────────────────────────────────────

@router.post("/compare", response_model=CompareResponse)
def compare_documents(
    request: CompareRequest,
    svc: DocumentService = Depends(get_document_service),
    user: dict = Depends(get_current_user),
):
    doc_a = _get_doc_or_404(request.document_id_a, svc, user)
    doc_b = _get_doc_or_404(request.document_id_b, svc, user)
    if doc_a.status != "ready":
        raise HTTPException(status_code=409, detail=f"Document A is not ready (status: {doc_a.status}).")
    if doc_b.status != "ready":
        raise HTTPException(status_code=409, detail=f"Document B is not ready (status: {doc_b.status}).")

    try:
        result = svc.compare_documents(doc_a, doc_b)
    except LLMError as e:
        raise HTTPException(status_code=429 if e.retry_after else 502, detail={"message": str(e), "retry_after": e.retry_after})

    return CompareResponse(
        document_id_a=request.document_id_a,
        document_id_b=request.document_id_b,
        **result,
    )


# ── List all documents ────────────────────────────────────────────────────────

@router.get("/documents", response_model=list[DocumentDetail])
def list_documents(
    svc: DocumentService = Depends(get_document_service),
    user: dict = Depends(get_current_user),
):
    # Admins see everything; regular users only their own documents.
    user_id = None if user.get("is_admin") else user.get("user_id")
    docs = svc.get_all_ready_documents(user_id=user_id)
    return [DocumentDetail.model_validate(d) for d in docs]


# ── Delete ────────────────────────────────────────────────────────────────────

@router.delete("/document/{doc_id}", status_code=204)
def delete_document(
    doc_id: str,
    svc: DocumentService = Depends(get_document_service),
    user: dict = Depends(get_current_user),
):
    doc = _get_doc_or_404(doc_id, svc, user)
    from pathlib import Path

    # Delete file
    try:
        Path(doc.file_path).unlink(missing_ok=True)
    except Exception:
        pass

    # Delete vector store collection
    delete_collection(doc_id)

    # Delete from DB (cascades to chat messages)
    svc.db.delete(doc)
    svc.db.commit()
