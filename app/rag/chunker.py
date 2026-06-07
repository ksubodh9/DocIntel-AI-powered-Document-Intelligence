import logging
"""
Text chunking with configurable size and overlap.
Chunks preserve page-number metadata for accurate citations.

Design notes:
- chunk_size=1000 chars is good for most documents; increase for dense technical text
- chunk_overlap=200 ensures context continuity across chunk boundaries
- Each chunk carries {page_number, chunk_index} for citation rendering
"""

from dataclasses import dataclass
from typing import Optional

from app.config.settings import get_settings

logger = logging.getLogger(__name__)
from app.utils.pdf_utils import PageContent

settings = get_settings()


@dataclass
class TextChunk:
    chunk_id: str           # "doc_{doc_id}_chunk_{index}"
    text: str
    page_number: int        # primary page for citation
    chunk_index: int
    char_start: int = 0
    char_end: int = 0
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.text.split())


def chunk_pages(
    pages: list[PageContent],
    doc_id: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[TextChunk]:
    """
    Chunk pages into overlapping text segments, preserving page metadata.

    Strategy:
    1. Concatenate page text with page boundary markers.
    2. Slide a window of chunk_size chars with chunk_overlap.
    3. Snap boundaries to word edges to avoid mid-word splits.
    4. Tag each chunk with the page number it started on.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    # Build a flat (char_offset -> page_number) mapping
    combined_text = ""
    page_boundaries: list[tuple[int, int]] = []  # (start_char, page_number)

    for page in pages:
        if not page.text.strip():
            continue
        start = len(combined_text)
        page_boundaries.append((start, page.page_number))
        combined_text += page.text + "\n\n"

    if not combined_text.strip():
        return []

    def page_at(char_offset: int) -> int:
        """Return the page number that owns char_offset."""
        page_num = 1
        for start, pnum in page_boundaries:
            if char_offset >= start:
                page_num = pnum
            else:
                break
        return page_num

    chunks: list[TextChunk] = []
    text_len = len(combined_text)
    start = 0
    index = 0

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Snap to word boundary (don't cut mid-word)
        if end < text_len and combined_text[end] not in (" ", "\n"):
            next_space = combined_text.find(" ", end)
            if next_space != -1 and (next_space - end) < 100:
                end = next_space

        chunk_text = combined_text[start:end].strip()
        if chunk_text:
            chunks.append(
                TextChunk(
                    chunk_id=f"doc_{doc_id}_chunk_{index}",
                    text=chunk_text,
                    page_number=page_at(start),
                    chunk_index=index,
                    char_start=start,
                    char_end=end,
                )
            )
            index += 1

        # Advance start with overlap.
        # If we just processed up to the end of the text, we're done.
        if end == text_len:
            break
        new_start = end - chunk_overlap
        if new_start <= start:
            start = end  # no forward progress — jump ahead to avoid infinite loop
        else:
            start = new_start

    logger.info(f"[Chunker] Produced {len(chunks)} chunks from {len(pages)} pages (size={chunk_size} overlap={chunk_overlap})")
    return chunks
