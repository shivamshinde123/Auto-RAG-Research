"""Tests for the MLflow experiment logger (run logging, error handling)."""

from unittest.mock import MagicMock, patch

from src.experiment_logger import ExperimentLogger


class TestExperimentLogger:
    def test_init_sets_experiment(self):
        with patch("src.experiment_logger.mlflow") as mock_mlflow:
            logger = ExperimentLogger("test_experiment")
            mock_mlflow.set_experiment.assert_called_with("test_experiment")

    def test_log_run_logs_params_and_metrics(self):
        with patch("src.experiment_logger.mlflow") as mock_mlflow:
            logger = ExperimentLogger("test_experiment")

            config = {"chunk_size": 512, "embedding_model": "all-MiniLM-L6-v2"}
            scores = {"faithfulness": 0.85, "composite_score": 0.80}
            logger.log_run(config=config, scores=scores, run_number=1, is_best=True)

            mock_mlflow.start_run.assert_called_once()
            mock_mlflow.log_param.assert_any_call("chunk_size", 512)
            mock_mlflow.log_metric.assert_any_call("faithfulness", 0.85)
            mock_mlflow.set_tag.assert_any_call("is_best", "True")

    def test_log_run_handles_mlflow_failure(self):
        with patch("src.experiment_logger.mlflow") as mock_mlflow:
            mock_mlflow.start_run.side_effect = RuntimeError("MLflow down")
            logger = ExperimentLogger("test_experiment")
            # Should not raise
            logger.log_run(
                config={"chunk_size": 512},
                scores={"composite_score": 0.8},
                run_number=1,
            )
