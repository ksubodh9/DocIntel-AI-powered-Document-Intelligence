"""
PDF utility functions: validation, text extraction, page-level chunking.
Uses PyMuPDF (fitz) as primary extractor and pdfplumber as fallback.
"""

import logging
import fitz  # PyMuPDF
import pdfplumber
import hashlib
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


@dataclass
class PageContent:
    page_number: int          # 1-indexed
    text: str
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.text.split())


@dataclass
class PDFContent:
    file_path: str
    filename: str
    page_count: int
    pages: list[PageContent] = field(default_factory=list)
    full_text: str = ""
    extraction_method: str = "pymupdf"
    is_scanned: bool = False
    file_hash: str = ""


class PDFValidationError(Exception):
    pass


def validate_pdf(file_path: Path, original_filename: str) -> None:
    """
    Validate that the file is a real, readable, non-empty PDF.
    Raises PDFValidationError with a user-friendly message on failure.
    """
    # Extension check
    if not original_filename.lower().endswith(".pdf"):
        raise PDFValidationError(f"Unsupported file type. Only PDF files are accepted.")

    # Size check
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise PDFValidationError(
            f"File exceeds the {settings.max_file_size_mb} MB size limit ({size_mb:.1f} MB uploaded)."
        )

    if file_path.stat().st_size == 0:
        raise PDFValidationError("The uploaded file is empty.")

    # Try opening with PyMuPDF to catch corruption
    try:
        doc = fitz.open(str(file_path))
        if doc.page_count == 0:
            raise PDFValidationError("The PDF contains no pages.")
        doc.close()
    except fitz.FileDataError:
        raise PDFValidationError("The PDF appears to be corrupted or not a valid PDF file.")
    except Exception as e:
        raise PDFValidationError(f"Could not open the PDF: {str(e)}")


def extract_text_pymupdf(file_path: Path) -> list[PageContent]:
    """Extract text page-by-page using PyMuPDF."""
    pages: list[PageContent] = []
    doc = fitz.open(str(file_path))
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text("text")
        text = clean_text(text)
        pages.append(PageContent(page_number=page_num + 1, text=text))
    doc.close()
    return pages


def extract_text_pdfplumber(file_path: Path) -> list[PageContent]:
    """Extract text using pdfplumber (better for tables/columns)."""
    pages: list[PageContent] = []
    with pdfplumber.open(str(file_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = clean_text(text)
            pages.append(PageContent(page_number=i + 1, text=text))
    return pages


def detect_scanned_pdf(pages: list[PageContent]) -> bool:
    """
    Heuristic: if > 70% of pages have very little text, the PDF is likely scanned.
    """
    if not pages:
        return False
    sparse = sum(1 for p in pages if p.word_count < 10)
    return (sparse / len(pages)) > 0.7


def extract_pdf(file_path: Path, filename: str) -> PDFContent:
    """
    Main extraction entry point.
    1. Try PyMuPDF
    2. If sparse text, try pdfplumber
    3. Detect if scanned (OCR not yet implemented – flag it)
    """
    # Compute hash for deduplication
    sha256 = hashlib.sha256(file_path.read_bytes()).hexdigest()

    logger.info(f"[PDFUtils] Trying PyMuPDF extraction on {file_path.name}")
    pages = extract_text_pymupdf(file_path)
    method = "pymupdf"

    total_words = sum(p.word_count for p in pages)
    if total_words < 50:
        logger.info("[PDFUtils] PyMuPDF text sparse - trying pdfplumber fallback...")
        plumber_pages = extract_text_pdfplumber(file_path)
        plumber_words = sum(p.word_count for p in plumber_pages)
        if plumber_words > total_words:
            pages = plumber_pages
            method = "pdfplumber"

    is_scanned = detect_scanned_pdf(pages)
    full_text = "\n\n".join(p.text for p in pages if p.text.strip())

    return PDFContent(
        file_path=str(file_path),
        filename=filename,
        page_count=len(pages),
        pages=pages,
        full_text=full_text,
        extraction_method=method,
        is_scanned=is_scanned,
        file_hash=sha256,
    )


def clean_text(text: str) -> str:
    """Remove control characters and normalize whitespace."""
    # Remove null bytes and other control chars (keep \n \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse 3+ consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse excessive spaces
    text = re.sub(r" {3,}", " ", text)
    return text.strip()
