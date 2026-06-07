"""
Unit tests for PDF utility functions.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import os

from app.utils.pdf_utils import clean_text, detect_scanned_pdf, PageContent, PDFValidationError


class TestCleanText:
    def test_removes_control_characters(self):
        text = "Hello\x00World\x01Test"
        result = clean_text(text)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "Hello" in result

    def test_collapses_multiple_newlines(self):
        text = "Line 1\n\n\n\n\nLine 2"
        result = clean_text(text)
        assert "\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result

    def test_collapses_excessive_spaces(self):
        text = "word1     word2"
        result = clean_text(text)
        assert "     " not in result

    def test_strips_whitespace(self):
        text = "   Hello World   "
        assert clean_text(text) == "Hello World"

    def test_preserves_normal_text(self):
        text = "This is a normal sentence.\nWith a newline."
        result = clean_text(text)
        assert "This is a normal sentence." in result


class TestDetectScannedPDF:
    def test_detects_scanned_when_mostly_empty(self):
        pages = [PageContent(page_number=i, text="a") for i in range(10)]  # 1 word each
        assert detect_scanned_pdf(pages) is True

    def test_not_scanned_when_text_rich(self):
        pages = [
            PageContent(page_number=i, text=" ".join(["word"] * 50))
            for i in range(10)
        ]
        assert detect_scanned_pdf(pages) is False

    def test_empty_pages_list(self):
        assert detect_scanned_pdf([]) is False

    def test_mixed_pages(self):
        # 3 sparse, 7 rich — should NOT be flagged as scanned
        pages = (
            [PageContent(page_number=i, text="a") for i in range(3)]
            + [PageContent(page_number=i + 3, text=" ".join(["word"] * 50)) for i in range(7)]
        )
        assert detect_scanned_pdf(pages) is False


class TestChunker:
    def test_basic_chunking(self):
        from app.rag.chunker import chunk_pages
        pages = [
            PageContent(page_number=1, text="word " * 300),
            PageContent(page_number=2, text="word " * 300),
        ]
        chunks = chunk_pages(pages, "test_doc", chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.chunk_id.startswith("doc_test_doc_chunk_")
            assert len(chunk.text) > 0
            assert chunk.page_number in (1, 2)

    def test_empty_pages(self):
        from app.rag.chunker import chunk_pages
        chunks = chunk_pages([], "test_doc")
        assert chunks == []

    def test_chunk_ids_are_unique(self):
        from app.rag.chunker import chunk_pages
        pages = [PageContent(page_number=1, text="word " * 500)]
        chunks = chunk_pages(pages, "test_doc", chunk_size=100, chunk_overlap=20)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_overlap_preserves_context(self):
        from app.rag.chunker import chunk_pages
        text = " ".join([f"word{i}" for i in range(200)])
        pages = [PageContent(page_number=1, text=text)]
        chunks = chunk_pages(pages, "doc", chunk_size=200, chunk_overlap=50)
        # With overlap, adjacent chunks should share some text
        if len(chunks) >= 2:
            end_of_first = chunks[0].text[-100:]
            start_of_second = chunks[1].text[:100]
            # At least some overlap should exist
            first_words = set(end_of_first.split())
            second_words = set(start_of_second.split())
            assert len(first_words & second_words) > 0
