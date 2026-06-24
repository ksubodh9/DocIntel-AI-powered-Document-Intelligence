"""
FastAPI route definitions.

Endpoints:
  POST   /upload                    Upload a PDF
  POST   /classify/{doc_id}         Classify the document
  POST   /summarize/{doc_id}        Generate summaries + topics
  POST   /metadata/{doc_id}         Extract document metadata
  GET    /tables/{doc_id}           Get extracted tables
  POST   /chat                      Chat with the document (RAG Q&A)
  GET    /session/mode              Post-login: what BYOK options to offer
  POST   /credentials/validate      Test a user-supplied provider + API key (BYOK)
  POST   /credentials               Save (encrypt) a provider key for the user
  GET    /credentials               List the user's saved keys (masked)
  DELETE /credentials/{provider}    Delete a saved key
  POST   /chat/multi                Chat across multiple documents
  POST   /compare                   Compare two documents
  GET    /document/{doc_id}         Get full document details
  GET    /document/{doc_id}/history Get chat history
  GET    /questions/{doc_id}        Get suggested questions
  GET    /documents                 List all ready documents
  DELETE /document/{doc_id}         Delete a document
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Request
from fastapi import status as http_status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.database.base import get_db
from app.services.llm_service import get_llm_service, LLMService, LLMError, LLMConfig, ALLOWED_PROVIDERS
from app.services.credential_service import CredentialService, UnsupportedProviderError
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
from app.schemas.credentials import (
    CredentialValidateRequest,
    CredentialValidateResponse,
    SaveCredentialRequest,
    StoredCredentialItem,
    StoredCredentialsResponse,
    SessionModeResponse,
)
from app.config.settings import get_settings

settings = get_settings()
router = APIRouter()
logger = logging.getLogger(__name__)


def _llm_http_error(e: LLMError) -> HTTPException:
    """Convert an LLMError into an HTTP error.

    The full technical detail goes to the server logs; the client only ever
    receives the safe, generic ``user_message`` (never provider/model/.env info).
    A rate-limit hint maps to 429, everything else to 502.
    """
    logger.warning("[LLM] %s", e)
    status_code = 429 if e.retry_after else 502
    return HTTPException(
        status_code=status_code,
        detail={"message": e.user_message, "retry_after": e.retry_after},
    )


def _resolve_llm_service(
    request: Request,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
) -> LLMService:
    """
    Per-request LLM service, resolved with this precedence:

      1. Session-only BYOK — X-LLM-Provider + X-LLM-Api-Key headers (key held in
         memory for this request only; never stored or logged).
      2. Saved BYOK — the user's active stored key (decrypted in memory).
      3. Server default — the configured .env keys.

    Send `X-LLM-Use-Default: true` to force the server default even when a saved
    key exists. An optional X-LLM-Model header overrides the model in mode 1.
    """
    use_default = request.headers.get("X-LLM-Use-Default", "").strip().lower() in ("1", "true", "yes")
    provider = request.headers.get("X-LLM-Provider")
    api_key = request.headers.get("X-LLM-Api-Key")
    model = request.headers.get("X-LLM-Model")

    # 1. Session-only header credentials
    if not use_default and provider and api_key:
        try:
            return LLMService(LLMConfig.for_byok(provider, api_key, model))
        except LLMError as e:
            raise HTTPException(status_code=400, detail={"message": e.user_message})

    # 2. Saved (persisted) credentials — active key for this user
    if not use_default:
        csvc = CredentialService(db)
        cred = csvc.get_active(user.get("user_id"))
        if cred is not None:
            try:
                key = csvc.decrypt_key(cred)
            except Exception:
                key = ""  # corrupt/unreadable ciphertext -> fall through to default
            if key:
                csvc.touch_last_used(cred)
                try:
                    return LLMService(LLMConfig.for_byok(cred.provider, key, cred.model))
                except LLMError:
                    pass  # stored provider no longer supported -> server default

    # 3. Server default
    return LLMService()


def _server_has_default_key() -> bool:
    """True if the server's configured primary provider has a usable key."""
    cfg = LLMConfig.from_settings(get_settings())
    if cfg.provider == "ollama":
        return True
    return bool(cfg.key_for(cfg.provider))


def _to_item(cred) -> StoredCredentialItem:
    last4 = cred.key_last4 or ""
    return StoredCredentialItem(
        provider=cred.provider,
        model=cred.model,
        masked_key=(f"\u2022\u2022\u2022\u2022{last4}" if last4 else "\u2022\u2022\u2022\u2022"),
        is_active=cred.is_active,
        created_at=cred.created_at,
        last_used_at=cred.last_used_at,
    )


def get_document_service(
    db: Session = Depends(get_db),
    llm: LLMService = Depends(_resolve_llm_service),
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
        raise _llm_http_error(e)

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
        raise _llm_http_error(e)

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
        raise _llm_http_error(e)
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
            raise _llm_http_error(e)

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
        raise _llm_http_error(e)
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
        raise _llm_http_error(e)

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
        raise _llm_http_error(e)

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



# ── BYOK: validate credentials ────────────────────────────────────────────────

@router.post("/credentials/validate", response_model=CredentialValidateResponse)
def validate_credentials(
    req: CredentialValidateRequest,
    user: dict = Depends(get_current_user),
):
    """
    Test a user-supplied provider + API key with one cheap call so the UI can
    give immediate feedback. The key is used in memory only and never stored or
    logged. The response never contains the key or any internal detail.
    """
    try:
        cfg = LLMConfig.for_byok(req.provider, req.api_key, req.model)
    except LLMError as e:
        return CredentialValidateResponse(valid=False, provider=req.provider, message=e.user_message)

    svc = LLMService(cfg)
    try:
        svc.complete("ping", max_tokens=4, temperature=0.0)
        return CredentialValidateResponse(valid=True, provider=cfg.provider, message="Credentials are valid.")
    except LLMError as e:
        return CredentialValidateResponse(valid=False, provider=cfg.provider, message=e.user_message)


# ── BYOK: post-login session mode ─────────────────────────────────────────────

@router.get("/session/mode", response_model=SessionModeResponse)
def session_mode(
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """
    Drives the post-login prompt. The client uses this to offer:
      * Continue        — use the server's default keys (if available).
      * Use my own key  — session-only and/or saved (if persistence available).
    """
    from app.utils.crypto import encryption_available

    active = CredentialService(db).get_active(user.get("user_id"))
    return SessionModeResponse(
        server_default_available=_server_has_default_key(),
        persistence_available=encryption_available(),
        has_saved_credentials=active is not None,
        active_provider=active.provider if active else None,
        supported_providers=sorted(ALLOWED_PROVIDERS),
    )


# ── BYOK: saved credentials (persisted) ───────────────────────────────────────

@router.post("/credentials", response_model=StoredCredentialItem, status_code=201)
def save_credential(
    req: SaveCredentialRequest,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """Encrypt and persist a provider key for the current user (optionally validated first)."""
    from app.utils.crypto import encryption_available, CredentialsEncryptionUnavailable

    if not encryption_available():
        raise HTTPException(
            status_code=503,
            detail={"message": "Saving keys is unavailable: server encryption is not configured. "
                               "You can still use session-only keys."},
        )

    try:
        cfg = LLMConfig.for_byok(req.provider, req.api_key, req.model)
    except LLMError as e:
        raise HTTPException(status_code=400, detail={"message": e.user_message})

    if req.validate_first:
        try:
            LLMService(cfg).complete("ping", max_tokens=4, temperature=0.0)
        except LLMError as e:
            raise HTTPException(status_code=400, detail={"message": f"Key was not saved. {e.user_message}"})

    try:
        cred = CredentialService(db).save(user.get("user_id"), req.provider, req.api_key, req.model)
    except UnsupportedProviderError:
        raise HTTPException(status_code=400, detail={"message": "That provider isn't supported."})
    except CredentialsEncryptionUnavailable:
        raise HTTPException(status_code=503, detail={"message": "Saving keys is unavailable right now."})

    return _to_item(cred)


@router.get("/credentials", response_model=StoredCredentialsResponse)
def list_credentials(
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """List the user's saved keys (masked — the real key is never returned)."""
    from app.utils.crypto import encryption_available

    items = [_to_item(c) for c in CredentialService(db).list_for_user(user.get("user_id"))]
    return StoredCredentialsResponse(encryption_available=encryption_available(), credentials=items)


@router.delete("/credentials/{provider}", status_code=204)
def delete_credential(
    provider: str,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """Revoke a saved key. After deletion the user falls back to server default (or another saved key if re-activated)."""
    if not CredentialService(db).delete(user.get("user_id"), provider):
        raise HTTPException(status_code=404, detail=f"No saved credential for provider '{provider}'.")
