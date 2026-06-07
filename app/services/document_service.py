"""
Document service: orchestrates PDF processing, RAG indexing, classification, and summarization.
Logs every step from file received to prompt sent to LLM.
"""

import uuid
import logging
import time
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.config.settings import get_settings
from app.models.document import Document, ChatMessage
from app.utils.pdf_utils import extract_pdf, validate_pdf, PDFValidationError
from app.rag.chunker import chunk_pages
from app.rag.vectorstore import index_chunks, retrieve_chunks, delete_collection
from app.services.llm_service import LLMService, LLMError
from app.prompts.templates import (
    CLASSIFICATION_PROMPT,
    SHORT_SUMMARY_PROMPT,
    DETAILED_SUMMARY_PROMPT,
    TOPICS_EXTRACTION_PROMPT,
    QA_SYSTEM_PROMPT,
    QA_USER_PROMPT,
    SUGGESTED_QUESTIONS_PROMPT,
)

settings = get_settings()
logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self, db: Session, llm: LLMService):
        self.db = db
        self.llm = llm

    # --------------------------------------------------------------------------
    # Upload
    # --------------------------------------------------------------------------

    def save_upload(self, file_bytes: bytes, original_filename: str) -> Document:
        t_start = time.perf_counter()
        doc_id = str(uuid.uuid4())
        logger.info(f"[Upload] START  file={original_filename}  size={len(file_bytes)/1024:.1f} KB  doc_id={doc_id}")

        upload_dir = settings.upload_dir
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = f"{doc_id}.pdf"
        file_path = upload_dir / safe_name
        file_path.write_bytes(file_bytes)
        logger.info(f"[Upload] File saved to disk: {file_path}")

        doc = Document(
            id=doc_id, filename=safe_name, original_filename=original_filename,
            file_path=str(file_path), file_size=len(file_bytes), status="processing",
        )
        self.db.add(doc)
        self.db.commit()
        logger.info(f"[Upload] DB record created (status=processing)")

        try:
            # Step 1: validate
            logger.info("[Upload] Step 1/4 - Validating PDF...")
            validate_pdf(file_path, original_filename)
            logger.info("[Upload] Validation passed")

            # Step 2: extract text
            logger.info("[Upload] Step 2/4 - Extracting text from PDF...")
            t_extract = time.perf_counter()
            pdf_content = extract_pdf(file_path, original_filename)
            logger.info(
                f"[Upload] Extraction done in {time.perf_counter()-t_extract:.1f}s  "
                f"pages={pdf_content.page_count}  "
                f"words={sum(p.word_count for p in pdf_content.pages)}  "
                f"method={pdf_content.extraction_method}  "
                f"scanned={pdf_content.is_scanned}"
            )
            doc.page_count = pdf_content.page_count

            if pdf_content.is_scanned:
                doc.status = "error"
                doc.error_message = "Scanned PDF detected. OCR not yet supported. Upload a text-based PDF."
                self.db.commit()
                logger.warning("[Upload] ABORTED - scanned PDF")
                return doc

            if not pdf_content.full_text.strip():
                doc.status = "error"
                doc.error_message = "Could not extract any text from this PDF."
                self.db.commit()
                logger.warning("[Upload] ABORTED - no text extracted")
                return doc

            logger.info(f"[Upload] Text preview: {pdf_content.full_text[:200].strip()!r}...")

            # Step 3: chunk
            logger.info("[Upload] Step 3/4 - Chunking text...")
            t_chunk = time.perf_counter()
            chunks = chunk_pages(pdf_content.pages, doc_id)
            logger.info(
                f"[Upload] Chunking done in {time.perf_counter()-t_chunk:.1f}s  "
                f"chunks={len(chunks)}  "
                f"chunk_size={settings.chunk_size}  overlap={settings.chunk_overlap}"
            )

            # Step 4: embed and index
            logger.info(f"[Upload] Step 4/4 - Embedding {len(chunks)} chunks and indexing in ChromaDB...")
            t_embed = time.perf_counter()
            collection_name = index_chunks(doc_id, chunks)
            logger.info(
                f"[Upload] Embedding + indexing done in {time.perf_counter()-t_embed:.1f}s  "
                f"collection={collection_name}"
            )

            doc.collection_name = collection_name
            doc.status = "ready"
            logger.info(f"[Upload] COMPLETE in {time.perf_counter()-t_start:.1f}s  status=ready")

        except PDFValidationError as e:
            doc.status = "error"
            doc.error_message = str(e)
            logger.error(f"[Upload] Validation error: {e}")
        except Exception as e:
            doc.status = "error"
            doc.error_message = f"Processing failed: {str(e)}"
            logger.exception(f"[Upload] Unexpected error: {e}")

        self.db.commit()
        self.db.refresh(doc)
        return doc

    # --------------------------------------------------------------------------
    # Classification
    # --------------------------------------------------------------------------

    def classify_document(self, doc: Document) -> Document:
        logger.info(f"[Classify] START doc_id={doc.id}")
        text_preview = self._get_text_preview(doc, max_chars=3000)
        logger.info(f"[Classify] Using first {len(text_preview)} chars for classification")
        prompt = CLASSIFICATION_PROMPT.format(text=text_preview)
        logger.info("[Classify] Sending classification prompt to LLM...")
        result = self.llm.complete_json(prompt)
        doc.document_type = result.get("document_type", "Other")
        doc.classification_confidence = float(result.get("confidence", 0.5))
        logger.info(f"[Classify] Result: type={doc.document_type}  confidence={doc.classification_confidence:.0%}")
        self.db.commit()
        self.db.refresh(doc)
        return doc

    # --------------------------------------------------------------------------
    # Summarization
    # --------------------------------------------------------------------------

    def summarize_document(self, doc: Document) -> Document:
        logger.info(f"[Summarize] START doc_id={doc.id}")
        full_text = self._get_full_text(doc)
        logger.info(f"[Summarize] Full text length: {len(full_text)} chars")
        text_for_summary = self._truncate_for_llm(full_text, max_chars=12000)
        logger.info(f"[Summarize] Truncated to {len(text_for_summary)} chars for LLM context")

        logger.info("[Summarize] Step 1/3 - Generating short summary...")
        doc.short_summary = self.llm.complete(SHORT_SUMMARY_PROMPT.format(text=text_for_summary), temperature=0.2)
        logger.info(f"[Summarize] Short summary: {doc.short_summary[:150]}...")

        logger.info("[Summarize] Step 2/3 - Generating detailed summary...")
        doc.detailed_summary = self.llm.complete(DETAILED_SUMMARY_PROMPT.format(text=text_for_summary), temperature=0.2)
        logger.info(f"[Summarize] Detailed summary: {doc.detailed_summary[:150]}...")

        logger.info("[Summarize] Step 3/3 - Extracting topics, keywords, entities...")
        topics_data = self.llm.complete_json(TOPICS_EXTRACTION_PROMPT.format(text=text_for_summary))
        doc.topics = topics_data.get("topics", [])
        doc.keywords = topics_data.get("keywords", [])
        doc.entities = topics_data.get("entities", [])
        logger.info(f"[Summarize] Topics: {doc.topics}  Keywords: {doc.keywords[:3]}  Entities: {doc.entities[:3]}")

        self.db.commit()
        self.db.refresh(doc)
        logger.info("[Summarize] COMPLETE")
        return doc

    # --------------------------------------------------------------------------
    # Suggested Questions
    # --------------------------------------------------------------------------

    def generate_suggested_questions(self, doc: Document) -> Document:
        logger.info(f"[Questions] Generating suggested questions for doc_id={doc.id}")
        summary = doc.short_summary or self._get_text_preview(doc, max_chars=1000)
        doc_type = doc.document_type or "Unknown"
        result = self.llm.complete_json(SUGGESTED_QUESTIONS_PROMPT.format(summary=summary, doc_type=doc_type))
        doc.suggested_questions = result.get("questions", [])
        logger.info(f"[Questions] Generated {len(doc.suggested_questions)} questions")
        self.db.commit()
        self.db.refresh(doc)
        return doc

    # --------------------------------------------------------------------------
    # Chat / Q&A
    # --------------------------------------------------------------------------

    def chat(self, doc: Document, user_message: str, include_history: bool = True) -> dict:
        logger.info(f"[Chat] START doc_id={doc.id}  question={user_message!r}")

        user_msg = ChatMessage(document_id=doc.id, role="user", content=user_message)
        self.db.add(user_msg)
        self.db.commit()

        logger.info(f"[Chat] Retrieving top-{settings.top_k_retrieval} chunks from ChromaDB...")
        t_retrieve = time.perf_counter()
        chunks = retrieve_chunks(doc.id, user_message, top_k=settings.top_k_retrieval)
        logger.info(
            f"[Chat] Retrieved {len(chunks)} chunks in {time.perf_counter()-t_retrieve:.2f}s  "
            f"scores={[round(c['relevance_score'],3) for c in chunks]}"
        )

        if not chunks:
            logger.info("[Chat] No relevant chunks found - returning default message")
            answer = "I could not find this information in the document."
            citations = []
        else:
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"[Chat] Chunk {i} (page={chunk['page_number']} score={chunk['relevance_score']:.3f}): {chunk['text'][:100]}...")

            context_parts = [f"[Chunk {i} | Page {c['page_number']}]\n{c['text']}" for i, c in enumerate(chunks, 1)]
            context = "\n\n---\n\n".join(context_parts)
            history = self._format_history(doc.id, last_n=6) if include_history else ""

            logger.info(f"[Chat] Building Q&A prompt (context={len(context)} chars)...")
            prompt = QA_USER_PROMPT.format(context=context, history=history, question=user_message)

            logger.info("[Chat] Sending Q&A prompt to LLM...")
            answer = self.llm.complete(prompt, system_prompt=QA_SYSTEM_PROMPT, temperature=0.1)
            logger.info(f"[Chat] Answer: {answer[:200]}...")

            citations = [
                {"page": c["page_number"], "text": c["text"][:300] + ("..." if len(c["text"]) > 300 else ""), "chunk_id": c["chunk_id"]}
                for c in chunks[:3]
            ]

        assistant_msg = ChatMessage(document_id=doc.id, role="assistant", content=answer, citations=citations)
        self.db.add(assistant_msg)
        self.db.commit()
        self.db.refresh(assistant_msg)
        logger.info(f"[Chat] COMPLETE  citations={len(citations)}")

        return {"message_id": assistant_msg.id, "answer": answer, "citations": citations, "sources_found": bool(chunks)}

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------

    def get_chat_history(self, doc_id: str) -> list:
        return (
            self.db.query(ChatMessage)
            .filter(ChatMessage.document_id == doc_id)
            .order_by(ChatMessage.created_at)
            .all()
        )

    def get_document(self, doc_id: str) -> Optional[Document]:
        return self.db.query(Document).filter(Document.id == doc_id).first()

    def _get_text_preview(self, doc: Document, max_chars: int = 3000) -> str:
        try:
            pdf_content = extract_pdf(Path(doc.file_path), doc.original_filename)
            return pdf_content.full_text[:max_chars]
        except Exception:
            return ""

    def _get_full_text(self, doc: Document) -> str:
        try:
            pdf_content = extract_pdf(Path(doc.file_path), doc.original_filename)
            return pdf_content.full_text
        except Exception:
            return ""

    def _truncate_for_llm(self, text: str, max_chars: int = 12000) -> str:
        if len(text) <= max_chars:
            return text
        head = text[:8000]
        tail = text[-(max_chars - 8000):]
        return head + "\n\n[... content truncated ...]\n\n" + tail

    def _format_history(self, doc_id: str, last_n: int = 6) -> str:
        messages = self.get_chat_history(doc_id)
        recent = messages[-(last_n * 2):]
        if not recent:
            return "No previous conversation."
        return "\n".join(f"{'User' if m.role == 'user' else 'Assistant'}: {m.content[:500]}" for m in recent)
