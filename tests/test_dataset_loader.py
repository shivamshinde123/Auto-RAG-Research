"""Tests for dataset_loader (PDF loading, deduplication, QA generation)."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.dataset_loader import _content_hash, load_documents


class TestContentHash:
    def test_same_content_same_hash(self):
        assert _content_hash("hello") == _content_hash("hello")

    def test_different_content_different_hash(self):
        assert _content_hash("hello") != _content_hash("world")


class TestLoadDocuments:
    @patch("src.dataset_loader._generate_qa_pairs")
    @patch("src.dataset_loader.get_data_source")
    def test_load_from_single_source(self, mock_get_ds, mock_gen_qa):
        """Loading from a single PDF source returns documents."""
        mock_source = MagicMock()
        mock_source.load.return_value = [
            Document(page_content="doc1", metadata={"source": "test"}),
            Document(page_content="doc2", metadata={"source": "test"}),
        ]
        mock_get_ds.return_value = mock_source
        mock_gen_qa.return_value = [{"question": "Q?", "ground_truth": "A"}]

        configs = [{"type": "local_pdf", "enabled": True, "path": "data/pdfs/"}]
        docs, qa_pairs = load_documents(configs)

        assert len(docs) == 2
        assert len(qa_pairs) == 1

    @patch("src.dataset_loader._generate_qa_pairs")
    @patch("src.dataset_loader.get_data_source")
    def test_deduplication(self, mock_get_ds, mock_gen_qa):
        """Duplicate documents are removed by content hash."""
        mock_source = MagicMock()
        mock_source.load.return_value = [
            Document(page_content="same content", metadata={"source": "a"}),
            Document(page_content="same content", metadata={"source": "b"}),
            Document(page_content="different", metadata={"source": "c"}),
        ]
        mock_get_ds.return_value = mock_source
        mock_gen_qa.return_value = []

        configs = [{"type": "local_pdf", "enabled": True}]
        docs, _ = load_documents(configs)
        assert len(docs) == 2

    @patch("src.dataset_loader._generate_qa_pairs")
    @patch("src.dataset_loader.get_data_source")
    def test_health_check_failure_skips_source(self, mock_get_ds, mock_gen_qa):
        """Sources that fail health check are skipped."""
        mock_source = MagicMock()
        mock_source.health_check.side_effect = ConnectionError("no access")
        mock_get_ds.return_value = mock_source
        mock_gen_qa.return_value = []

        configs = [{"type": "local_pdf", "enabled": True}]
        docs, _ = load_documents(configs)
        assert len(docs) == 0

    def test_no_enabled_sources(self):
        """Returns empty when no sources are enabled."""
        configs = [{"type": "local_pdf", "enabled": False}]
        docs, qa_pairs = load_documents(configs)
        assert docs == []
        assert qa_pairs == []

    def test_empty_configs(self):
        """Returns empty when no configs provided."""
        docs, qa_pairs = load_documents([])
        assert docs == []
        assert qa_pairs == []
