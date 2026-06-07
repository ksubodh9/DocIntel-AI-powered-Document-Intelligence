"""
Document service: orchestrates PDF processing, RAG indexing, classification,
summarization, metadata extraction, table extraction, and chat (single + multi-doc).
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
from app.rag.chunker import chunk_pages, TextChunk
from app.rag.vectorstore import index_chunks, retrieve_chunks, retrieve_chunks_multi, delete_collection
from app.services.llm_service import LLMService, LLMError
from app.prompts.templates import (
    CLASSIFICATION_PROMPT,
    SHORT_SUMMARY_PROMPT,
    DETAILED_SUMMARY_PROMPT,
    TOPICS_EXTRACTION_PROMPT,
    METADATA_EXTRACTION_PROMPT,
    COMPARISON_PROMPT,
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
        logger.info("[Upload] DB record created (status=processing)")

        try:
            # Step 1: validate
            logger.info("[Upload] Step 1/5 - Validating PDF...")
            validate_pdf(file_path, original_filename)
            logger.info("[Upload] Validation passed")

            # Step 2: extract text + tables + metadata
            logger.info("[Upload] Step 2/5 - Extracting text, tables, and metadata...")
            t_extract = time.perf_counter()
            pdf_content = extract_pdf(file_path, original_filename)
            logger.info(
                f"[Upload] Extraction done in {time.perf_counter()-t_extract:.1f}s  "
                f"pages={pdf_content.page_count}  "
                f"words={sum(p.word_count for p in pdf_content.pages)}  "
                f"method={pdf_content.extraction_method}  "
                f"scanned={pdf_content.is_scanned}  "
                f"tables={len(pdf_content.tables)}"
            )
            doc.page_count = pdf_content.page_count

            # Cache full_text in DB so downstream calls never re-parse the file
            doc.full_text = pdf_content.full_text

            # Store native PDF metadata
            doc.doc_metadata = pdf_content.native_metadata

            # Store extracted tables
            if pdf_content.tables:
                doc.has_tables = True
                doc.table_count = len(pdf_content.tables)
                doc.tables = [
                    {"page": t.page_number, "caption": t.caption, "markdown": t.markdown}
                    for t in pdf_content.tables
                ]
            else:
                doc.has_tables = False
                doc.table_count = 0

            if pdf_content.is_scanned and pdf_content.extraction_method != "ocr":
                doc.status = "error"
                doc.error_message = (
                    "Scanned PDF detected and OCR is not available. "
                    "Install pytesseract to process scanned PDFs."
                )
                self.db.commit()
                logger.warning("[Upload] ABORTED - scanned PDF, OCR unavailable")
                return doc

            if not pdf_content.full_text.strip():
                doc.status = "error"
                doc.error_message = "Could not extract any text from this PDF."
                self.db.commit()
                logger.warning("[Upload] ABORTED - no text extracted")
                return doc

            logger.info(f"[Upload] Text preview: {pdf_content.full_text[:200].strip()!r}...")

            # Step 3: chunk text
            logger.info("[Upload] Step 3/5 - Chunking text...")
            t_chunk = time.perf_counter()
            chunks = chunk_pages(pdf_content.pages, doc_id)

            # Step 3b: also create chunks for tables so they're queryable via RAG
            if pdf_content.tables:
                table_chunks = _make_table_chunks(pdf_content.tables, doc_id, len(chunks))
                chunks.extend(table_chunks)
                logger.info(f"[Upload] Added {len(table_chunks)} table chunk(s)")

            logger.info(
                f"[Upload] Chunking done in {time.perf_counter()-t_chunk:.1f}s  "
                f"chunks={len(chunks)}"
            )

            # Step 4: embed + index
            logger.info(f"[Upload] Step 4/5 - Embedding {len(chunks)} chunks...")
            t_embed = time.perf_counter()
            collection_name = index_chunks(doc_id, chunks)
            logger.info(
                f"[Upload] Embedding + indexing done in {time.perf_counter()-t_embed:.1f}s  "
                f"collection={collection_name}"
            )
            doc.collection_name = collection_name

            # Step 5: word count in metadata
            total_words = sum(p.word_count for p in pdf_content.pages)
            doc.doc_metadata = {**(doc.doc_metadata or {}), "word_count": total_words}

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
        logger.info(f"[Classify] Using first {len(text_preview)} chars")
        result = self.llm.complete_json(CLASSIFICATION_PROMPT.format(text=text_preview))
        doc.document_type = result.get("document_type", "Other")
        doc.classification_confidence = float(result.get("confidence", 0.5))
        logger.info(f"[Classify] type={doc.document_type}  confidence={doc.classification_confidence:.0%}")
        self.db.commit()
        self.db.refresh(doc)
        return doc

    # --------------------------------------------------------------------------
    # Metadata Extraction
    # --------------------------------------------------------------------------

    def extract_metadata(self, doc: Document) -> Document:
        logger.info(f"[Metadata] START doc_id={doc.id}")
        text_preview = self._get_text_preview(doc, max_chars=3000)
        llm_meta = self.llm.complete_json(METADATA_EXTRACTION_PROMPT.format(text=text_preview))
        # Merge native PDF metadata (already in doc.doc_metadata) with LLM-extracted
        existing = doc.doc_metadata or {}
        merged = {**existing, **{k: v for k, v in llm_meta.items() if v is not None}}
        doc.doc_metadata = merged
        logger.info(f"[Metadata] Extracted: {list(merged.keys())}")
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

        logger.info("[Summarize] Step 1/3 - Short summary...")
        doc.short_summary = self.llm.complete(
            SHORT_SUMMARY_PROMPT.format(text=text_for_summary), temperature=0.2
        )

        logger.info("[Summarize] Step 2/3 - Detailed summary...")
        doc.detailed_summary = self.llm.complete(
            DETAILED_SUMMARY_PROMPT.format(text=text_for_summary), temperature=0.2
        )

        logger.info("[Summarize] Step 3/3 - Topics/keywords/entities...")
        topics_data = self.llm.complete_json(TOPICS_EXTRACTION_PROMPT.format(text=text_for_summary))
        doc.topics = topics_data.get("topics", [])
        doc.keywords = topics_data.get("keywords", [])
        doc.entities = topics_data.get("entities", [])

        self.db.commit()
        self.db.refresh(doc)
        logger.info("[Summarize] COMPLETE")
        return doc

    # --------------------------------------------------------------------------
    # Suggested Questions
    # --------------------------------------------------------------------------

    def generate_suggested_questions(self, doc: Document) -> Document:
        logger.info(f"[Questions] doc_id={doc.id}")
        summary = doc.short_summary or self._get_text_preview(doc, max_chars=1000)
        doc_type = doc.document_type or "Unknown"
        result = self.llm.complete_json(
            SUGGESTED_QUESTIONS_PROMPT.format(summary=summary, doc_type=doc_type)
        )
        doc.suggested_questions = result.get("questions", [])
        self.db.commit()
        self.db.refresh(doc)
        return doc

    # --------------------------------------------------------------------------
    # Document Comparison
    # --------------------------------------------------------------------------

    def compare_documents(self, doc_a: Document, doc_b: Document) -> dict:
        logger.info(f"[Compare] doc_a={doc_a.id} doc_b={doc_b.id}")
        # Ensure both have summaries; generate on-the-fly if missing
        if not doc_a.short_summary:
            doc_a = self.summarize_document(doc_a)
        if not doc_b.short_summary:
            doc_b = self.summarize_document(doc_b)

        prompt = COMPARISON_PROMPT.format(
            title_a=doc_a.original_filename,
            type_a=doc_a.document_type or "Unknown",
            summary_a=doc_a.short_summary,
            topics_a=", ".join(doc_a.topics or []),
            title_b=doc_b.original_filename,
            type_b=doc_b.document_type or "Unknown",
            summary_b=doc_b.short_summary,
            topics_b=", ".join(doc_b.topics or []),
        )
        result = self.llm.complete_json(prompt)
        logger.info("[Compare] COMPLETE")
        return {
            "similarities": result.get("similarities", []),
            "differences": result.get("differences", []),
            "recommendation": result.get("recommendation", ""),
            "detailed_comparison": result.get("detailed_comparison", ""),
        }

    # --------------------------------------------------------------------------
    # Chat (single document)
    # --------------------------------------------------------------------------

    def chat(self, doc: Document, user_message: str, include_history: bool = True) -> dict:
        logger.info(f"[Chat] doc_id={doc.id}  question={user_message!r}")

        user_msg = ChatMessage(document_id=doc.id, role="user", content=user_message)
        self.db.add(user_msg)
        self.db.commit()

        t_retrieve = time.perf_counter()
        chunks = retrieve_chunks(doc.id, user_message, top_k=settings.top_k_retrieval)
        logger.info(
            f"[Chat] Retrieved {len(chunks)} chunks in {time.perf_counter()-t_retrieve:.2f}s  "
            f"scores={[round(c['relevance_score'],3) for c in chunks]}"
        )

        answer, citations = self._generate_answer(chunks, user_message, doc.id, include_history)

        assistant_msg = ChatMessage(document_id=doc.id, role="assistant", content=answer, citations=citations)
        self.db.add(assistant_msg)
        self.db.commit()
        self.db.refresh(assistant_msg)

        return {"message_id": assistant_msg.id, "answer": answer, "citations": citations, "sources_found": bool(chunks)}

    # --------------------------------------------------------------------------
    # Multi-document Chat
    # --------------------------------------------------------------------------

    def chat_multi(self, doc_ids: list[str], doc_map: dict, user_message: str) -> dict:
        """
        Chat across multiple documents. doc_map is {doc_id: Document}.
        """
        logger.info(f"[MultiChat] docs={doc_ids}  question={user_message!r}")

        t_retrieve = time.perf_counter()
        chunks = retrieve_chunks_multi(doc_ids, user_message, top_k=settings.top_k_retrieval)
        logger.info(f"[MultiChat] Retrieved {len(chunks)} chunks in {time.perf_counter()-t_retrieve:.2f}s")

        # Enrich chunks with document names
        for c in chunks:
            did = c.get("doc_id", "")
            c["document_name"] = doc_map.get(did, {}).original_filename if did in doc_map else did

        # Build context with document attribution
        context_parts = []
        for i, c in enumerate(chunks, 1):
            doc_name = c.get("document_name", "Unknown")
            context_parts.append(f"[Chunk {i} | {doc_name} | Page {c['page_number']}]\n{c['text']}")
        context = "\n\n---\n\n".join(context_parts) if context_parts else ""

        if not chunks:
            answer = "I could not find relevant information across the selected documents."
            citations = []
        else:
            prompt = QA_USER_PROMPT.format(context=context, history="", question=user_message)
            answer = self.llm.complete(prompt, system_prompt=QA_SYSTEM_PROMPT, temperature=0.1)
            citations = [
                {
                    "page": c["page_number"],
                    "text": c["text"][:300] + ("..." if len(c["text"]) > 300 else ""),
                    "chunk_id": c["chunk_id"],
                    "relevance_score": c["relevance_score"],
                    "document_id": c.get("doc_id", ""),
                    "document_name": c.get("document_name", ""),
                }
                for c in chunks[:3]
            ]

        msg_id = str(uuid.uuid4())
        return {"message_id": msg_id, "answer": answer, "citations": citations, "sources_found": bool(chunks)}

    # --------------------------------------------------------------------------
    # CRUD helpers
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

    def get_all_ready_documents(self) -> list[Document]:
        return self.db.query(Document).filter(Document.status == "ready").all()

    # --------------------------------------------------------------------------
    # Private helpers
    # --------------------------------------------------------------------------

    def _generate_answer(self, chunks, user_message, doc_id, include_history):
        if not chunks:
            return "I could not find this information in the document.", []

        for i, c in enumerate(chunks, 1):
            logger.info(f"[Chat] Chunk {i} (page={c['page_number']} score={c['relevance_score']:.3f}): {c['text'][:100]}...")

        context_parts = [f"[Chunk {i} | Page {c['page_number']}]\n{c['text']}" for i, c in enumerate(chunks, 1)]
        context = "\n\n---\n\n".join(context_parts)
        history = self._format_history(doc_id, last_n=6) if include_history else ""

        prompt = QA_USER_PROMPT.format(context=context, history=history, question=user_message)
        answer = self.llm.complete(prompt, system_prompt=QA_SYSTEM_PROMPT, temperature=0.1)

        citations = [
            {
                "page": c["page_number"],
                "text": c["text"][:300] + ("..." if len(c["text"]) > 300 else ""),
                "chunk_id": c["chunk_id"],
                "relevance_score": round(c["relevance_score"], 3),
            }
            for c in chunks[:3]
        ]
        return answer, citations

    def _get_full_text(self, doc: Document) -> str:
        # Use cached full_text from DB — never re-parse the file
        if doc.full_text:
            return doc.full_text
        # Fallback: re-parse (only for documents uploaded before this version)
        logger.warning(f"[Service] full_text not cached for doc {doc.id} — re-parsing")
        try:
            pdf_content = extract_pdf(Path(doc.file_path), doc.original_filename)
            doc.full_text = pdf_content.full_text
            self.db.commit()
            return pdf_content.full_text
        except Exception as e:
            logger.error(f"[Service] Failed to re-parse PDF: {e}")
            return ""

    def _get_text_preview(self, doc: Document, max_chars: int = 3000) -> str:
        return self._get_full_text(doc)[:max_chars]

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
        return "\n".join(
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content[:500]}" for m in recent
        )


# --------------------------------------------------------------------------
# Helper: make RAG chunks from extracted tables
# --------------------------------------------------------------------------

def _make_table_chunks(tables, doc_id: str, start_index: int) -> list[TextChunk]:
    """Wrap each extracted table as a TextChunk so it's indexed in ChromaDB."""
    chunks = []
    for i, table in enumerate(tables):
        text = f"{table.caption}\n\n{table.markdown}"
        chunks.append(
            TextChunk(
                chunk_id=f"doc_{doc_id}_table_{i}",
                text=text,
                page_number=table.page_number,
                chunk_index=start_index + i,
            )
        )
    return chunks
