"""Tests for local_pdf data source connector."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data_sources.local_pdf import LocalPdfDataSource


@pytest.fixture
def pdf_dir(tmp_path):
    """Create a temp dir with a dummy PDF file."""
    # Create a minimal valid PDF using PyMuPDF
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
    def test_loads_pages(self, source):
        docs = source.load()
        assert len(docs) == 2
        assert "Hello from page 1" in docs[0].page_content
        assert "Hello from page 2" in docs[1].page_content

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
