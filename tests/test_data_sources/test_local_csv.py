"""Tests for local_csv data source connector."""

import pytest

from src.data_sources.local_csv import LocalCsvDataSource


def _write_csv(path, content):
    """Helper to write a CSV file."""
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def csv_dir(tmp_path):
    """Create a temp dir with a sample CSV file."""
    _write_csv(
        tmp_path / "data.csv",
        "id,text,label\n1,Hello world,pos\n2,Some notes here,neg\n",
    )
    return tmp_path


@pytest.fixture
def source(csv_dir):
    return LocalCsvDataSource(
        {"type": "local_csv", "path": str(csv_dir), "text_column": "text"}
    )


class TestValidateConfig:
    def test_valid_config(self, source):
        assert source.validate_config() is True

    def test_missing_path(self):
        src = LocalCsvDataSource({"type": "local_csv", "text_column": "text"})
        with pytest.raises(ValueError, match="missing required 'path'"):
            src.validate_config()

    def test_missing_text_column(self, tmp_path):
        src = LocalCsvDataSource({"type": "local_csv", "path": str(tmp_path)})
        with pytest.raises(ValueError, match="missing required 'text_column'"):
            src.validate_config()


class TestHealthCheck:
    def test_healthy(self, source):
        assert source.health_check() is True

    def test_nonexistent_dir(self, tmp_path):
        src = LocalCsvDataSource(
            {"type": "local_csv", "path": str(tmp_path / "nope"), "text_column": "text"}
        )
        with pytest.raises(RuntimeError, match="does not exist"):
            src.health_check()

    def test_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        src = LocalCsvDataSource(
            {"type": "local_csv", "path": str(empty), "text_column": "text"}
        )
        with pytest.raises(RuntimeError, match="No .csv files found"):
            src.health_check()

    def test_not_a_directory(self, tmp_path):
        file = tmp_path / "notadir.csv"
        file.write_text("a,b\n1,2\n")
        src = LocalCsvDataSource(
            {"type": "local_csv", "path": str(file), "text_column": "a"}
        )
        with pytest.raises(RuntimeError, match="not a directory"):
            src.health_check()

    def test_missing_text_column_in_header(self, tmp_path):
        _write_csv(tmp_path / "bad.csv", "id,name\n1,Alice\n")
        src = LocalCsvDataSource(
            {"type": "local_csv", "path": str(tmp_path), "text_column": "text"}
        )
        with pytest.raises(RuntimeError, match="text_column 'text' not found"):
            src.health_check()


class TestLoad:
    def test_loads_rows(self, source):
        docs = source.load()
        assert len(docs) == 2
        contents = {d.page_content for d in docs}
        assert "Hello world" in contents
        assert "Some notes here" in contents

    def test_metadata(self, source):
        docs = source.load()
        doc = docs[0]
        assert doc.metadata["source_type"] == "local_csv"
        assert doc.metadata["file_name"] == "data.csv"
        assert doc.metadata["row_index"] == 0

    def test_skips_empty_rows(self, tmp_path):
        _write_csv(
            tmp_path / "gaps.csv",
            "id,text\n1,Hello\n2,\n3,  \n4,World\n",
        )
        src = LocalCsvDataSource(
            {"type": "local_csv", "path": str(tmp_path), "text_column": "text"}
        )
        docs = src.load()
        assert len(docs) == 2
        contents = {d.page_content for d in docs}
        assert "Hello" in contents
        assert "World" in contents

    def test_empty_dir_returns_empty(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        src = LocalCsvDataSource(
            {"type": "local_csv", "path": str(empty), "text_column": "text"}
        )
        assert src.load() == []

    def test_multiple_csv_files(self, tmp_path):
        _write_csv(tmp_path / "a.csv", "text\nFirst\n")
        _write_csv(tmp_path / "b.csv", "text\nSecond\n")
        src = LocalCsvDataSource(
            {"type": "local_csv", "path": str(tmp_path), "text_column": "text"}
        )
        docs = src.load()
        assert len(docs) == 2
        contents = {d.page_content for d in docs}
        assert "First" in contents
        assert "Second" in contents

    def test_row_index_is_per_file(self, tmp_path):
        _write_csv(tmp_path / "multi.csv", "text\nA\nB\nC\n")
        src = LocalCsvDataSource(
            {"type": "local_csv", "path": str(tmp_path), "text_column": "text"}
        )
        docs = src.load()
        indices = [d.metadata["row_index"] for d in docs]
        assert indices == [0, 1, 2]
