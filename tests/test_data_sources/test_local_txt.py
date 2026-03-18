"""Tests for local_txt data source connector."""

import pytest
from pathlib import Path

from src.data_sources.local_txt import LocalTxtDataSource


@pytest.fixture
def txt_dir(tmp_path):
    """Create a temp dir with sample text files."""
    (tmp_path / "hello.txt").write_text("Hello world", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Some notes here", encoding="utf-8")
    return tmp_path


@pytest.fixture
def source(txt_dir):
    return LocalTxtDataSource({"type": "local_txt", "path": str(txt_dir)})


class TestValidateConfig:
    def test_valid_config(self, source):
        assert source.validate_config() is True

    def test_missing_path(self):
        src = LocalTxtDataSource({"type": "local_txt"})
        with pytest.raises(ValueError, match="missing required 'path'"):
            src.validate_config()


class TestHealthCheck:
    def test_healthy(self, source):
        assert source.health_check() is True

    def test_nonexistent_dir(self, tmp_path):
        src = LocalTxtDataSource({"type": "local_txt", "path": str(tmp_path / "nope")})
        with pytest.raises(RuntimeError, match="does not exist"):
            src.health_check()

    def test_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        src = LocalTxtDataSource({"type": "local_txt", "path": str(empty)})
        with pytest.raises(RuntimeError, match="No .txt files found"):
            src.health_check()

    def test_not_a_directory(self, tmp_path):
        file = tmp_path / "notadir.txt"
        file.write_text("hi")
        src = LocalTxtDataSource({"type": "local_txt", "path": str(file)})
        with pytest.raises(RuntimeError, match="not a directory"):
            src.health_check()


class TestLoad:
    def test_loads_files(self, source):
        docs = source.load()
        assert len(docs) == 2
        contents = {d.page_content for d in docs}
        assert "Hello world" in contents
        assert "Some notes here" in contents

    def test_metadata(self, source):
        docs = source.load()
        doc = next(d for d in docs if d.metadata["file_name"] == "hello.txt")
        assert doc.metadata["source_type"] == "local_txt"
        assert doc.metadata["char_count"] == len("Hello world")

    def test_empty_dir_returns_empty(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        src = LocalTxtDataSource({"type": "local_txt", "path": str(empty)})
        assert src.load() == []

    def test_skips_whitespace_only_files(self, tmp_path):
        (tmp_path / "blank.txt").write_text("   \n\n  ", encoding="utf-8")
        src = LocalTxtDataSource({"type": "local_txt", "path": str(tmp_path)})
        assert src.load() == []

    def test_detects_encoding(self, tmp_path):
        """Files with non-UTF-8 encoding are read without crashing."""
        # Use a longer string so chardet has enough signal to detect encoding.
        text = "Le café est très bon. Ça coûte cinq euros pour un résumé complet."
        (tmp_path / "latin.txt").write_bytes(text.encode("latin-1"))
        src = LocalTxtDataSource({"type": "local_txt", "path": str(tmp_path)})
        docs = src.load()
        assert len(docs) == 1
        # The file should be readable (not raise) and contain recognizable content
        assert "caf" in docs[0].page_content
