"""Tests for dataset_loader module."""

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
    @patch("src.dataset_loader.get_data_source")
    def test_load_from_single_source(self, mock_get_ds):
        mock_source = MagicMock()
        mock_source.load.return_value = [
            Document(page_content="doc1", metadata={"source": "test"}),
            Document(page_content="doc2", metadata={"source": "test"}),
        ]
        mock_get_ds.return_value = mock_source

        configs = [{"type": "local_txt", "enabled": True, "path": "/data"}]
        docs, qa_pairs = load_documents(configs)

        assert len(docs) == 2
        assert qa_pairs is None

    @patch("src.dataset_loader.get_data_source")
    def test_load_from_multiple_sources(self, mock_get_ds):
        mock_source1 = MagicMock()
        mock_source1.load.return_value = [
            Document(page_content="doc1", metadata={"source": "src1"}),
        ]
        mock_source2 = MagicMock()
        mock_source2.load.return_value = [
            Document(page_content="doc2", metadata={"source": "src2"}),
        ]
        mock_get_ds.side_effect = [mock_source1, mock_source2]

        configs = [
            {"type": "local_txt", "enabled": True},
            {"type": "local_csv", "enabled": True},
        ]
        docs, _ = load_documents(configs)
        assert len(docs) == 2

    @patch("src.dataset_loader.get_data_source")
    def test_deduplication(self, mock_get_ds):
        mock_source = MagicMock()
        mock_source.load.return_value = [
            Document(page_content="same content", metadata={"source": "a"}),
            Document(page_content="same content", metadata={"source": "b"}),
            Document(page_content="different", metadata={"source": "c"}),
        ]
        mock_get_ds.return_value = mock_source

        configs = [{"type": "local_txt", "enabled": True}]
        docs, _ = load_documents(configs)
        assert len(docs) == 2

    @patch("src.dataset_loader.get_data_source")
    def test_graceful_failure_one_source(self, mock_get_ds):
        mock_source1 = MagicMock()
        mock_source1.load.side_effect = RuntimeError("source1 crashed")
        mock_source2 = MagicMock()
        mock_source2.load.return_value = [
            Document(page_content="doc", metadata={"source": "ok"}),
        ]
        mock_get_ds.side_effect = [mock_source1, mock_source2]

        configs = [
            {"type": "bad", "enabled": True},
            {"type": "good", "enabled": True},
        ]
        docs, _ = load_documents(configs)
        assert len(docs) == 1

    @patch("src.dataset_loader.get_data_source")
    def test_health_check_failure_skips_source(self, mock_get_ds):
        mock_source = MagicMock()
        mock_source.health_check.side_effect = ConnectionError("no access")
        mock_get_ds.return_value = mock_source

        configs = [{"type": "bad", "enabled": True}]
        docs, _ = load_documents(configs)
        assert len(docs) == 0

    def test_no_enabled_sources(self):
        configs = [{"type": "local_txt", "enabled": False}]
        docs, qa_pairs = load_documents(configs)
        assert docs == []
        assert qa_pairs is None

    def test_empty_configs(self):
        docs, qa_pairs = load_documents([])
        assert docs == []
        assert qa_pairs is None

    @patch("src.dataset_loader.get_data_source")
    def test_huggingface_qa_pairs(self, mock_get_ds):
        mock_source = MagicMock()
        mock_source.load_with_qa.return_value = (
            [Document(page_content="ctx", metadata={"source": "hf"})],
            [{"question": "Q?", "ground_truth": "A"}],
        )
        mock_get_ds.return_value = mock_source

        configs = [{"type": "huggingface", "enabled": True}]
        docs, qa_pairs = load_documents(configs)
        assert len(docs) == 1
        assert len(qa_pairs) == 1
        assert qa_pairs[0]["question"] == "Q?"
