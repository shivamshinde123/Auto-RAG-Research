"""Tests for the local PDF connector (config validation, health check, parsing, fallback)."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data_sources.local_pdf import LocalPdfDataSource

# Check if a PDF parser is available for tests that need actual parsing
_has_pdf_parser = False
try:
    import fitz
    _has_pdf_parser = True
except ImportError:
    try:
        import pdfplumber
        _has_pdf_parser = True
    except ImportError:
        pass

requires_pdf_parser = pytest.mark.skipif(
    not _has_pdf_parser, reason="No PDF parser (fitz or pdfplumber) available"
)


@pytest.fixture
def pdf_dir(tmp_path):
    """Create a temp dir with a dummy PDF file."""
    try:
        import fitz
        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        text_point = fitz.Point(72, 72)
        page.insert_text(text_point, "Hello from page 1")
        page2 = doc.new_page()
        page2.insert_text(text_point, "Hello from page 2")
        doc.save(str(pdf_path))
        doc.close()
    except ImportError:
        # Fallback: write a minimal valid PDF manually
        pdf_path = tmp_path / "test.pdf"
        pdf_content = (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R 4 0 R]/Count 2>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
            b"/Contents 5 0 R/Resources<</Font<</F1 7 0 R>>>>>>endobj\n"
            b"4 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
            b"/Contents 6 0 R/Resources<</Font<</F1 7 0 R>>>>>>endobj\n"
            b"5 0 obj<</Length 44>>stream\nBT /F1 12 Tf 72 720 Td (Hello from page 1) Tj ET\nendstream\nendobj\n"
            b"6 0 obj<</Length 44>>stream\nBT /F1 12 Tf 72 720 Td (Hello from page 2) Tj ET\nendstream\nendobj\n"
            b"7 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
            b"xref\n0 8\n"
            b"trailer<</Size 8/Root 1 0 R>>\nstartxref\n0\n%%EOF\n"
        )
        pdf_path.write_bytes(pdf_content)
    return tmp_path


@pytest.fixture
def source(pdf_dir):
    return LocalPdfDataSource({"type": "local_pdf", "path": str(pdf_dir)})


class TestValidateConfig:
    def test_valid_config(self, source):
        assert source.validate_config() is True

    def test_missing_path(self):
        src = LocalPdfDataSource({"type": "local_pdf"})
        with pytest.raises(ValueError, match="missing required 'path'"):
            src.validate_config()


class TestHealthCheck:
    def test_healthy(self, source):
        assert source.health_check() is True

    def test_nonexistent_dir(self, tmp_path):
        src = LocalPdfDataSource({"type": "local_pdf", "path": str(tmp_path / "nope")})
        with pytest.raises(RuntimeError, match="does not exist"):
            src.health_check()

    def test_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        src = LocalPdfDataSource({"type": "local_pdf", "path": str(empty)})
        with pytest.raises(RuntimeError, match="No PDF files found"):
            src.health_check()

    def test_not_a_directory(self, tmp_path):
        file = tmp_path / "notadir.txt"
        file.write_text("hi")
        src = LocalPdfDataSource({"type": "local_pdf", "path": str(file)})
        with pytest.raises(RuntimeError, match="not a directory"):
            src.health_check()


class TestLoad:
    @requires_pdf_parser
    def test_loads_pages(self, source):
        docs = source.load()
        assert len(docs) == 2
        assert "Hello from page 1" in docs[0].page_content
        assert "Hello from page 2" in docs[1].page_content

    @requires_pdf_parser
    def test_metadata(self, source):
        docs = source.load()
        meta = docs[0].metadata
        assert meta["source_type"] == "local_pdf"
        assert meta["file_name"] == "test.pdf"
        assert meta["page_number"] == 1
        assert meta["total_pages"] == 2

    def test_empty_dir_returns_empty(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        src = LocalPdfDataSource({"type": "local_pdf", "path": str(empty)})
        assert src.load() == []

    def test_fallback_to_pdfplumber(self, source, pdf_dir):
        """If PyMuPDF fails, pdfplumber is used as fallback."""
        with patch.object(
            source, "_load_with_pymupdf", side_effect=Exception("pymupdf broke")
        ) as mock_pymupdf:
            with patch.object(
                source, "_load_with_pdfplumber", return_value=[]
            ) as mock_plumber:
                source.load()
                mock_pymupdf.assert_called()
                mock_plumber.assert_called()

    def test_both_parsers_fail_skips_file(self, source):
        """If both parsers fail, the file is skipped with no crash."""
        with patch.object(
            source, "_load_with_pymupdf", side_effect=Exception("fail1")
        ):
            with patch.object(
                source, "_load_with_pdfplumber", side_effect=Exception("fail2")
            ):
                docs = source.load()
                assert docs == []
