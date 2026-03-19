"""Tests for evaluator module."""

import math
import sys
from unittest.mock import MagicMock, patch

import pytest

from src.evaluator import METRIC_NAMES, compute_composite


class TestComputeComposite:
    def test_equal_weights(self):
        scores = {
            "faithfulness": 0.8,
            "answer_relevancy": 0.6,
            "context_precision": 0.7,
            "context_recall": 0.9,
        }
        result = compute_composite(scores)
        assert abs(result - 0.75) < 1e-6

    def test_custom_weights(self):
        scores = {
            "faithfulness": 1.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }
        weights = {
            "faithfulness": 4.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }
        result = compute_composite(scores, weights)
        assert abs(result - 1.0) < 1e-6

    def test_nan_values_ignored(self):
        scores = {
            "faithfulness": 0.8,
            "answer_relevancy": float("nan"),
            "context_precision": 0.6,
            "context_recall": float("nan"),
        }
        result = compute_composite(scores)
        assert abs(result - 0.7) < 1e-6

    def test_all_nan_returns_nan(self):
        scores = {name: float("nan") for name in METRIC_NAMES}
        result = compute_composite(scores)
        assert math.isnan(result)

    def test_missing_metrics_treated_as_nan(self):
        scores = {"faithfulness": 0.8}
        result = compute_composite(scores)
        assert abs(result - 0.8) < 1e-6


class TestEvaluate:
    def _setup_mocks(self):
        """Set up mock modules for ragas and datasets."""
        mock_ragas = MagicMock()
        mock_datasets = MagicMock()
        mock_ragas_metrics = MagicMock()
        return mock_ragas, mock_datasets, mock_ragas_metrics

    def test_evaluate_returns_all_metrics(self):
        mock_ragas = MagicMock()
        mock_ragas.evaluate.return_value = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.78,
            "context_precision": 0.82,
            "context_recall": 0.90,
        }
        mock_datasets = MagicMock()

        with patch.dict(sys.modules, {
            "ragas": mock_ragas,
            "ragas.metrics": MagicMock(),
            "datasets": mock_datasets,
        }):
            # Re-import to pick up mocked modules
            from src.evaluator import evaluate

            results = [
                {
                    "question": "What is X?",
                    "answer": "X is Y",
                    "contexts": ["context about X"],
                    "ground_truth": "X is Y",
                }
            ]

            scores = evaluate(results)
            assert "faithfulness" in scores
            assert "answer_relevancy" in scores
            assert "context_precision" in scores
            assert "context_recall" in scores
            assert "composite_score" in scores
            assert abs(scores["composite_score"] - 0.8375) < 1e-4

    def test_evaluate_handles_ragas_failure(self):
        mock_ragas = MagicMock()
        mock_ragas.evaluate.side_effect = RuntimeError("RAGAS crashed")
        mock_datasets = MagicMock()

        with patch.dict(sys.modules, {
            "ragas": mock_ragas,
            "ragas.metrics": MagicMock(),
            "datasets": mock_datasets,
        }):
            from src.evaluator import evaluate

            results = [
                {
                    "question": "Q",
                    "answer": "A",
                    "contexts": ["c"],
                    "ground_truth": "A",
                }
            ]

            scores = evaluate(results)
            for name in METRIC_NAMES:
                assert math.isnan(scores[name])
            assert math.isnan(scores["composite_score"])

    def test_evaluate_handles_partial_metric_failure(self):
        mock_ragas = MagicMock()
        mock_ragas.evaluate.return_value = {
            "faithfulness": 0.80,
            "answer_relevancy": None,
            "context_precision": 0.70,
        }
        mock_datasets = MagicMock()

        with patch.dict(sys.modules, {
            "ragas": mock_ragas,
            "ragas.metrics": MagicMock(),
            "datasets": mock_datasets,
        }):
            from src.evaluator import evaluate

            results = [
                {
                    "question": "Q",
                    "answer": "A",
                    "contexts": ["c"],
                    "ground_truth": "A",
                }
            ]

            scores = evaluate(results)
            assert scores["faithfulness"] == 0.80
            assert math.isnan(scores["answer_relevancy"])
            assert scores["context_precision"] == 0.70
            assert math.isnan(scores["context_recall"])
            assert abs(scores["composite_score"] - 0.75) < 1e-6
