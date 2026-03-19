"""MLflow experiment logger for tracking RAG pipeline optimization runs."""

import json
import logging
import tempfile
from pathlib import Path

import mlflow

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Logs experiment runs to MLflow."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self._setup_experiment()

    def _setup_experiment(self):
        """Create or get the MLflow experiment."""
        try:
            mlflow.set_experiment(self.experiment_name)
            logger.info("MLflow experiment: %s", self.experiment_name)
        except Exception as e:
            logger.warning("Failed to set up MLflow experiment: %s", e)

    def log_run(
        self,
        config: dict,
        scores: dict,
        run_number: int,
        is_best: bool = False,
        reasoning: str | None = None,
    ):
        """Log a single experiment run to MLflow.

        Args:
            config: Hyperparameter config dict.
            scores: RAGAS scores dict.
            run_number: Iteration number.
            is_best: Whether this is a new best score.
            reasoning: Optional agent reasoning text to log as artifact.
        """
        try:
            with mlflow.start_run(run_name=f"iteration_{run_number}"):
                # Log parameters
                for key, value in config.items():
                    mlflow.log_param(key, value)

                # Log metrics
                for key, value in scores.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)

                # Log tags
                mlflow.set_tag("run_number", str(run_number))
                mlflow.set_tag("is_best", str(is_best))

                # Log config as artifact
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    json.dump(config, f, indent=2)
                    f.flush()
                    mlflow.log_artifact(f.name, "config")

                # Log reasoning if provided
                if reasoning:
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".md", delete=False
                    ) as f:
                        f.write(reasoning)
                        f.flush()
                        mlflow.log_artifact(f.name, "reasoning")

            logger.info(
                "Logged run %d to MLflow (is_best=%s)", run_number, is_best
            )
        except Exception as e:
            logger.warning("Failed to log run %d to MLflow: %s", run_number, e)
