"""
PDF utility functions: validation, text extraction, table extraction, OCR, metadata.
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
class TableData:
    page_number: int
    caption: str
    markdown: str


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
    native_metadata: dict = field(default_factory=dict)
    tables: list[TableData] = field(default_factory=list)


class PDFValidationError(Exception):
    pass


def validate_pdf(file_path: Path, original_filename: str) -> None:
    if not original_filename.lower().endswith(".pdf"):
        raise PDFValidationError("Unsupported file type. Only PDF files are accepted.")
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise PDFValidationError(
            f"File exceeds the {settings.max_file_size_mb} MB size limit ({size_mb:.1f} MB uploaded)."
        )
    if file_path.stat().st_size == 0:
        raise PDFValidationError("The uploaded file is empty.")
    try:
        doc = fitz.open(str(file_path))
        if doc.page_count == 0:
            raise PDFValidationError("The PDF contains no pages.")
        doc.close()
    except fitz.FileDataError:
        raise PDFValidationError("The PDF appears to be corrupted or not a valid PDF file.")
    except PDFValidationError:
        raise
    except Exception as e:
        raise PDFValidationError(f"Could not open the PDF: {str(e)}")


def extract_native_metadata(file_path: Path) -> dict:
    """Pull author/title/creation-date from PDF metadata headers."""
    try:
        doc = fitz.open(str(file_path))
        raw = doc.metadata or {}
        doc.close()
        return {
            "pdf_title": raw.get("title", "") or None,
            "pdf_author": raw.get("author", "") or None,
            "pdf_subject": raw.get("subject", "") or None,
            "pdf_creator": raw.get("creator", "") or None,
            "pdf_creation_date": raw.get("creationDate", "") or None,
        }
    except Exception:
        return {}


def extract_text_pymupdf(file_path: Path) -> list[PageContent]:
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
    pages: list[PageContent] = []
    with pdfplumber.open(str(file_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = clean_text(text)
            pages.append(PageContent(page_number=i + 1, text=text))
    return pages


def extract_tables_pdfplumber(file_path: Path) -> list[TableData]:
    """
    Extract all tables from the PDF using pdfplumber.
    Returns each table as a markdown string with its page number.
    """
    tables: list[TableData] = []
    try:
        with pdfplumber.open(str(file_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                if not page_tables:
                    continue
                for t_idx, raw_table in enumerate(page_tables):
                    if not raw_table or len(raw_table) < 2:
                        continue
                    md = _table_to_markdown(raw_table)
                    if not md:
                        continue
                    caption = f"Table {len(tables) + 1} (page {i + 1})"
                    tables.append(TableData(page_number=i + 1, caption=caption, markdown=md))
    except Exception as e:
        logger.warning(f"[PDFUtils] Table extraction failed: {e}")
    return tables


def _table_to_markdown(table: list[list]) -> str:
    """Convert a pdfplumber table (list of rows) to markdown."""
    # Clean cells
    rows = []
    for row in table:
        cleaned = [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row]
        rows.append(cleaned)
    if not rows:
        return ""
    # First row as header
    header = rows[0]
    divider = ["---"] * len(header)
    body = rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(divider) + " |",
    ]
    for row in body:
        # Pad short rows
        while len(row) < len(header):
            row.append("")
        lines.append("| " + " | ".join(row[:len(header)]) + " |")
    return "\n".join(lines)


def ocr_page_pymupdf(page) -> str:
    """
    Render a single PyMuPDF page to an image and run pytesseract OCR.
    Falls back to empty string if pytesseract is not installed.
    """
    try:
        import pytesseract
        from PIL import Image
        import io
        mat = fitz.Matrix(2.0, 2.0)  # 2x scale for better OCR accuracy
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        text = pytesseract.image_to_string(img)
        return clean_text(text)
    except ImportError:
        logger.warning("[PDFUtils] pytesseract/Pillow not installed — OCR skipped")
        return ""
    except Exception as e:
        logger.warning(f"[PDFUtils] OCR failed on page: {e}")
        return ""


def extract_text_ocr(file_path: Path) -> list[PageContent]:
    """OCR all pages of a scanned PDF using pytesseract."""
    pages: list[PageContent] = []
    doc = fitz.open(str(file_path))
    logger.info(f"[PDFUtils] Running OCR on {doc.page_count} pages...")
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = ocr_page_pymupdf(page)
        pages.append(PageContent(page_number=page_num + 1, text=text))
        logger.info(f"[PDFUtils] OCR page {page_num + 1}/{doc.page_count}: {len(text.split())} words")
    doc.close()
    return pages


def detect_scanned_pdf(pages: list[PageContent]) -> bool:
    """Heuristic: if >70% of pages have <10 words, PDF is likely scanned."""
    if not pages:
        return False
    sparse = sum(1 for p in pages if p.word_count < 10)
    return (sparse / len(pages)) > 0.7


def extract_pdf(file_path: Path, filename: str) -> PDFContent:
    """
    Main extraction entry point.
    1. Try PyMuPDF
    2. If sparse text, try pdfplumber
    3. If still scanned, try OCR (pytesseract)
    4. Extract tables with pdfplumber
    5. Pull native PDF metadata
    """
    sha256 = hashlib.sha256(file_path.read_bytes()).hexdigest()
    native_meta = extract_native_metadata(file_path)

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
            total_words = plumber_words

    is_scanned = detect_scanned_pdf(pages)
    if is_scanned:
        logger.info("[PDFUtils] Scanned PDF detected — attempting OCR...")
        ocr_pages = extract_text_ocr(file_path)
        ocr_words = sum(p.word_count for p in ocr_pages)
        if ocr_words > total_words:
            pages = ocr_pages
            method = "ocr"
            is_scanned = detect_scanned_pdf(pages)  # re-check after OCR
            logger.info(f"[PDFUtils] OCR extracted {ocr_words} words")
        else:
            logger.warning("[PDFUtils] OCR produced no usable text")

    # Extract tables
    logger.info("[PDFUtils] Extracting tables...")
    tables = extract_tables_pdfplumber(file_path)
    if tables:
        logger.info(f"[PDFUtils] Found {len(tables)} table(s)")

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
        native_metadata=native_meta,
        tables=tables,
    )


def clean_text(text: str) -> str:
    """Remove control characters and normalize whitespace."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {3,}", " ", text)
    return text.strip()
