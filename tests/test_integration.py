"""End-to-end integration test for the full experiment loop.

Uses mocks for all external APIs (OpenAI, RAGAS) so it runs without
API keys. Tests the full flow: config -> load docs -> run pipeline ->
evaluate -> agent suggests next config -> repeat.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@pytest.fixture
def test_config_path(tmp_path):
    """Create a minimal program.md for testing."""
    config = tmp_path / "program_test.md"
    config.write_text(
        """\
## Data Sources

[[data_sources]]
type: local_pdf
path: data/pdfs/
enabled: true

## QA Generation
num_qa_pairs: 5

## Search Space
chunk_size: [512]
chunk_overlap: [50]
top_k: [3]
embedding_model: [all-MiniLM-L6-v2]
llm_model: [gpt-4o-mini]

## Optimization Target
primary_metric: context_recall
secondary_metric: faithfulness
min_threshold: 0.95

## Constraints
max_iterations: 2
max_cost_usd: 1.0

## Experiment
experiment_name: test_integration
git_checkpoints: false
""",
        encoding="utf-8",
    )
    return str(config)


class TestIntegrationMocked:
    """Integration test using mocks for external APIs (runs without API keys)."""

    def test_full_loop_mocked(self, test_config_path, tmp_path, monkeypatch):
        """Test the full experiment flow with mocked external calls."""
        monkeypatch.chdir(tmp_path)

        from langchain_core.documents import Document

        mock_docs = [
            Document(page_content=f"Context about topic {i}.", metadata={"source": "test.pdf", "source_type": "local_pdf"})
            for i in range(5)
        ]
        mock_qa_pairs = [
            {"question": f"What is topic {i}?", "ground_truth": f"Topic {i} is about testing."}
            for i in range(5)
        ]

        with (
            patch("src.dataset_loader.get_data_source") as mock_get_ds,
            patch("src.dataset_loader._generate_qa_pairs") as mock_gen_qa,
            patch("src.rag_pipeline.build_vector_store") as mock_build_vs,
            patch("src.rag_pipeline.get_llm") as mock_get_llm,
            patch("src.evaluator.evaluate") as mock_evaluate,
            patch("src.experiment_logger.ExperimentLogger") as mock_logger_cls,
        ):
            # Mock data source
            mock_source = MagicMock()
            mock_source.load.return_value = mock_docs
            mock_get_ds.return_value = mock_source

            # Mock QA generation
            mock_gen_qa.return_value = mock_qa_pairs

            # Mock vector store
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = [Document(page_content="retrieved context")]
            mock_vs = MagicMock()
            mock_vs.as_retriever.return_value = mock_retriever
            mock_build_vs.return_value = mock_vs

            # Mock LLM
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(content="mocked answer")
            mock_get_llm.return_value = mock_llm

            # Mock evaluator -- return scores above threshold on 2nd iteration
            mock_evaluate.side_effect = [
                {
                    "faithfulness": 0.80,
                    "answer_relevancy": 0.75,
                    "context_precision": 0.70,
                    "context_recall": 0.65,
                    "composite_score": 0.725,
                },
                {
                    "faithfulness": 0.95,
                    "answer_relevancy": 0.96,
                    "context_precision": 0.97,
                    "context_recall": 0.98,
                    "composite_score": 0.965,
                },
            ]

            # Mock experiment logger
            mock_exp_logger = MagicMock()
            mock_logger_cls.return_value = mock_exp_logger

            # Mock the agent for iteration 2
            with patch("src.agent.OpenAI", create=True) as mock_openai:
                import sys

                mock_openai_module = MagicMock()
                mock_client = MagicMock()
                mock_openai_module.OpenAI.return_value = mock_client
                next_config = {
                    "chunk_size": 512,
                    "chunk_overlap": 50,
                    "top_k": 3,
                    "embedding_model": "all-MiniLM-L6-v2",
                    "llm_model": "gpt-4o-mini",
                }
                agent_response = MagicMock()
                agent_response.choices = [
                    MagicMock(message=MagicMock(content=json.dumps({
                        "analysis": "Test analysis",
                        "decision": "Test decision",
                        "config": next_config,
                    })))
                ]
                mock_client.chat.completions.create.return_value = agent_response

                with patch.dict(sys.modules, {"openai": mock_openai_module}):
                    from main import run_experiment
                    run_experiment(test_config_path)

            # Verify experiment history was created
            history_path = tmp_path / "experiment_history.jsonl"
            assert history_path.exists()
            lines = [l for l in history_path.read_text().splitlines() if l.strip()]
            assert len(lines) >= 1

            # Verify MLflow logger was called
            assert mock_exp_logger.log_run.call_count >= 1
