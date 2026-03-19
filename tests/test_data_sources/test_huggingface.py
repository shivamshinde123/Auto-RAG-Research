"""Tests for HuggingFace data source connector."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.data_sources.huggingface import HuggingFaceDataSource


class TestHuggingFaceValidateConfig:
    def test_missing_dataset_name(self):
        source = HuggingFaceDataSource({"type": "huggingface"})
        with pytest.raises(ValueError, match="dataset_name"):
            source.validate_config()

    def test_missing_split(self):
        source = HuggingFaceDataSource({"type": "huggingface", "dataset_name": "squad"})
        with pytest.raises(ValueError, match="split"):
            source.validate_config()

    def test_valid_config(self):
        source = HuggingFaceDataSource({
            "type": "huggingface", "dataset_name": "squad", "split": "validation"
        })
        assert source.validate_config() is True


class TestHuggingFaceLoadWithQa:
    def test_squad_extraction(self):
        mock_datasets = MagicMock()
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=3)
        mock_ds.__iter__ = MagicMock(return_value=iter([
            {"context": "Paris is the capital of France.", "question": "What is the capital of France?", "answers": {"text": ["Paris"]}},
            {"context": "Paris is the capital of France.", "question": "Where is the Eiffel Tower?", "answers": {"text": ["Paris"]}},
            {"context": "Berlin is the capital of Germany.", "question": "What is the capital of Germany?", "answers": {"text": ["Berlin"]}},
        ]))
        mock_ds.select = MagicMock(return_value=mock_ds)
        mock_datasets.load_dataset.return_value = mock_ds

        with patch.dict(sys.modules, {"datasets": mock_datasets}):
            source = HuggingFaceDataSource({
                "type": "huggingface", "dataset_name": "squad", "split": "validation"
            })
            docs, qa_pairs = source.load_with_qa()

        # 2 unique contexts
        assert len(docs) == 2
        # 3 QA pairs
        assert len(qa_pairs) == 3
        assert qa_pairs[0]["question"] == "What is the capital of France?"
        assert qa_pairs[0]["ground_truth"] == "Paris"

    def test_sample_size_limit(self):
        mock_datasets = MagicMock()
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=100)

        # After select, return 5 items
        selected_ds = MagicMock()
        selected_ds.__len__ = MagicMock(return_value=5)
        selected_ds.__iter__ = MagicMock(return_value=iter([
            {"context": f"ctx{i}", "question": f"q{i}", "answers": {"text": [f"a{i}"]}}
            for i in range(5)
        ]))
        mock_ds.select.return_value = selected_ds
        mock_datasets.load_dataset.return_value = mock_ds

        with patch.dict(sys.modules, {"datasets": mock_datasets}):
            source = HuggingFaceDataSource({
                "type": "huggingface", "dataset_name": "squad",
                "split": "validation", "sample_size": 5,
            })
            docs, qa_pairs = source.load_with_qa()

        mock_ds.select.assert_called_once_with(range(5))
        assert len(qa_pairs) == 5

    def test_load_returns_only_documents(self):
        mock_datasets = MagicMock()
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=1)
        mock_ds.__iter__ = MagicMock(return_value=iter([
            {"context": "test", "question": "q?", "answers": {"text": ["a"]}},
        ]))
        mock_datasets.load_dataset.return_value = mock_ds

        with patch.dict(sys.modules, {"datasets": mock_datasets}):
            source = HuggingFaceDataSource({
                "type": "huggingface", "dataset_name": "squad", "split": "validation"
            })
            docs = source.load()

        assert len(docs) == 1
        assert isinstance(docs, list)


class TestHuggingFaceHealthCheck:
    def test_health_check_passes(self):
        mock_datasets = MagicMock()

        with patch.dict(sys.modules, {"datasets": mock_datasets}):
            source = HuggingFaceDataSource({
                "type": "huggingface", "dataset_name": "squad", "split": "validation"
            })
            assert source.health_check() is True

    def test_health_check_fails(self):
        mock_datasets = MagicMock()
        mock_datasets.load_dataset_builder.side_effect = Exception("not found")

        with patch.dict(sys.modules, {"datasets": mock_datasets}):
            source = HuggingFaceDataSource({
                "type": "huggingface", "dataset_name": "nonexistent", "split": "train"
            })
            with pytest.raises(ConnectionError, match="Cannot access"):
                source.health_check()
