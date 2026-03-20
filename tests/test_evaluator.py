"""Tests for the RAGAS evaluator (composite scoring, metric aggregation, error handling)."""

import math
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.evaluator import METRIC_NAMES, _RAGAS_METRIC_NAME_MAP, compute_composite


def _mock_ragas_result(metric_values: dict):
    """Create a mock RAGAS EvaluationResult with .to_pandas() support.

    Args:
        metric_values: dict mapping canonical metric names to values.
            Uses _RAGAS_METRIC_NAME_MAP to translate to RAGAS column names.
    """
    columns = {}
    for canonical_name, value in metric_values.items():
        ragas_name = _RAGAS_METRIC_NAME_MAP.get(canonical_name, canonical_name)
        columns[ragas_name] = [value]
    df = pd.DataFrame(columns)
    result = MagicMock()
    result.to_pandas.return_value = df
    return result


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
        mock_result = _mock_ragas_result({
            "faithfulness": 0.85,
            "answer_relevancy": 0.78,
            "context_precision": 0.82,
            "context_recall": 0.90,
        })

        # Mock RAGAS modules that are imported inside evaluate()
        mock_ragas = MagicMock()
        mock_ragas.evaluate.return_value = mock_result
        mock_ragas.EvaluationDataset = MagicMock()
        mock_ragas.SingleTurnSample = MagicMock()
        mock_ragas_metrics = MagicMock()
        mock_langchain_openai = MagicMock()

        with patch.dict(sys.modules, {
            "ragas": mock_ragas,
            "ragas.metrics": mock_ragas_metrics,
            "langchain_openai": mock_langchain_openai,
        }):
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
        mock_ragas.EvaluationDataset = MagicMock()
        mock_ragas.SingleTurnSample = MagicMock()
        mock_ragas_metrics = MagicMock()
        mock_langchain_openai = MagicMock()

        with patch.dict(sys.modules, {
            "ragas": mock_ragas,
            "ragas.metrics": mock_ragas_metrics,
            "langchain_openai": mock_langchain_openai,
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
        # Only faithfulness and context_precision have values;
        # answer_relevancy is None and context_recall is missing
        mock_result = _mock_ragas_result({
            "faithfulness": 0.80,
            "answer_relevancy": None,
            "context_precision": 0.70,
        })

        mock_ragas = MagicMock()
        mock_ragas.evaluate.return_value = mock_result
        mock_ragas.EvaluationDataset = MagicMock()
        mock_ragas.SingleTurnSample = MagicMock()
        mock_ragas_metrics = MagicMock()
        mock_langchain_openai = MagicMock()

        with patch.dict(sys.modules, {
            "ragas": mock_ragas,
            "ragas.metrics": mock_ragas_metrics,
            "langchain_openai": mock_langchain_openai,
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
