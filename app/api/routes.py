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

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from fastapi import status as http_status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.database.base import get_db
from app.services.llm_service import get_llm_service, LLMService, LLMError
from app.services.document_service import DocumentService
from app.rag.vectorstore import delete_collection
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


def _get_doc_or_404(doc_id: str, svc: DocumentService):
    doc = svc.get_document(doc_id)
    if not doc:
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
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    svc: DocumentService = Depends(get_document_service),
):
    """
    Upload a PDF document.
    The file is validated, text is extracted, and chunks are indexed in ChromaDB.
    """
    # Validate content type
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported. Please upload a .pdf file.",
            )

    # Read file bytes (size check happens inside service)
    file_bytes = await file.read()
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_file_size_mb} MB.",
        )

    # save_upload does blocking work (disk write, PDF parse, embedding) -
    # run it in a worker thread so the event loop (and /health) stays responsive.
    doc = await run_in_threadpool(svc.save_upload, file_bytes, file.filename or "document.pdf")

    return UploadResponse(
        document_id=doc.id,
        filename=doc.original_filename,
        file_size=doc.file_size,
        page_count=doc.page_count,
        status=doc.status,
        message=(
            "PDF uploaded and indexed successfully."
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
):
    doc = _get_doc_or_404(doc_id, svc)
    _require_ready(doc)

    try:
        doc = svc.classify_document(doc)
    except LLMError as e:
        raise HTTPException(status_code=429 if e.retry_after else 502, detail={"message": str(e), "retry_after": e.retry_after})

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
):
    doc = _get_doc_or_404(doc_id, svc)
    _require_ready(doc)

    try:
        doc = svc.summarize_document(doc)
        if not doc.suggested_questions:
            doc = svc.generate_suggested_questions(doc)
    except LLMError as e:
        raise HTTPException(status_code=429 if e.retry_after else 502, detail={"message": str(e), "retry_after": e.retry_after})

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
):
    doc = _get_doc_or_404(request.document_id, svc)
    _require_ready(doc)

    try:
        result = svc.chat(doc, request.message, request.include_history)
    except LLMError as e:
        raise HTTPException(status_code=429 if e.retry_after else 502, detail={"message": str(e), "retry_after": e.retry_after})

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
):
    doc = _get_doc_or_404(doc_id, svc)
    return DocumentDetail.model_validate(doc)


# ── Chat History ──────────────────────────────────────────────────────────────

@router.get("/document/{doc_id}/history", response_model=ChatHistoryResponse)
def get_chat_history(
    doc_id: str,
    svc: DocumentService = Depends(get_document_service),
):
    _get_doc_or_404(doc_id, svc)
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
):
    doc = _get_doc_or_404(doc_id, svc)
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
):
    doc = _get_doc_or_404(doc_id, svc)
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
):
    doc = _get_doc_or_404(doc_id, svc)
    _require_ready(doc)
    raw_tables = doc.tables or []
    tables = [TableItem(page=t["page"], caption=t["caption"], markdown=t["markdown"]) for t in raw_tables]
    return TablesResponse(document_id=doc.id, table_count=len(tables), tables=tables)


# ── Multi-document Chat ───────────────────────────────────────────────────────

@router.post("/chat/multi", response_model=MultiChatResponse)
def multi_chat(
    request: MultiChatRequest,
    svc: DocumentService = Depends(get_document_service),
):
    # Validate all documents exist and are ready
    doc_map = {}
    for doc_id in request.document_ids:
        doc = svc.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
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
):
    doc_a = svc.get_document(request.document_id_a)
    doc_b = svc.get_document(request.document_id_b)
    if not doc_a:
        raise HTTPException(status_code=404, detail=f"Document A '{request.document_id_a}' not found.")
    if not doc_b:
        raise HTTPException(status_code=404, detail=f"Document B '{request.document_id_b}' not found.")
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
):
    docs = svc.get_all_ready_documents()
    return [DocumentDetail.model_validate(d) for d in docs]


# ── Delete ────────────────────────────────────────────────────────────────────

@router.delete("/document/{doc_id}", status_code=204)
def delete_document(
    doc_id: str,
    svc: DocumentService = Depends(get_document_service),
):
    doc = _get_doc_or_404(doc_id, svc)
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
