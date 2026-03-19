"""Tests for agent module."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent import (
    _is_duplicate,
    _load_history,
    _validate_config,
)

SEARCH_SPACE = {
    "chunk_size": [256, 512, 1024],
    "chunk_overlap": [25, 50, 100],
    "top_k": [3, 5, 10],
    "embedding_model": ["all-MiniLM-L6-v2", "BGE-large", "text-embedding-ada-002"],
    "llm_model": ["gpt-4o-mini", "gpt-3.5-turbo"],
}

SAMPLE_CONFIG = {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "top_k": 5,
    "embedding_model": "all-MiniLM-L6-v2",
    "llm_model": "gpt-4o-mini",
}


class TestLoadHistory:
    def test_load_empty_file(self, tmp_path):
        path = tmp_path / "history.jsonl"
        path.write_text("")
        assert _load_history(path) == []

    def test_load_nonexistent_file(self, tmp_path):
        path = tmp_path / "nonexistent.jsonl"
        assert _load_history(path) == []

    def test_load_valid_history(self, tmp_path):
        path = tmp_path / "history.jsonl"
        entries = [
            {"iteration": 1, "config": SAMPLE_CONFIG, "composite_score": 0.75},
            {"iteration": 2, "config": SAMPLE_CONFIG, "composite_score": 0.80},
        ]
        path.write_text("\n".join(json.dumps(e) for e in entries))
        result = _load_history(path)
        assert len(result) == 2
        assert result[0]["iteration"] == 1


class TestValidateConfig:
    def test_valid_config(self):
        assert _validate_config(SAMPLE_CONFIG, SEARCH_SPACE) is True

    def test_invalid_chunk_size(self):
        config = {**SAMPLE_CONFIG, "chunk_size": 999}
        with pytest.raises(ValueError, match="chunk_size=999 not in allowed"):
            _validate_config(config, SEARCH_SPACE)

    def test_invalid_embedding_model(self):
        config = {**SAMPLE_CONFIG, "embedding_model": "nonexistent"}
        with pytest.raises(ValueError, match="embedding_model"):
            _validate_config(config, SEARCH_SPACE)


class TestIsDuplicate:
    def test_not_duplicate(self):
        history = [{"config": SAMPLE_CONFIG}]
        different = {**SAMPLE_CONFIG, "chunk_size": 256}
        assert _is_duplicate(different, history) is False

    def test_is_duplicate(self):
        history = [{"config": SAMPLE_CONFIG}]
        assert _is_duplicate(SAMPLE_CONFIG, history) is True

    def test_empty_history(self):
        assert _is_duplicate(SAMPLE_CONFIG, []) is False


def _make_mock_openai(config_to_return):
    """Create a mock openai module with a client that returns the given config."""
    mock_openai_module = MagicMock()
    mock_client = MagicMock()
    mock_openai_module.OpenAI.return_value = mock_client

    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps({
                    "analysis": "Context recall is weakest.",
                    "decision": "Increase top_k and use BGE-large.",
                    "config": config_to_return,
                })
            )
        )
    ]
    mock_client.chat.completions.create.return_value = response
    return mock_openai_module, mock_client


class TestSuggestNextConfig:
    def test_suggest_writes_config_and_notes(self, tmp_path):
        new_config = {
            "chunk_size": 256,
            "chunk_overlap": 50,
            "top_k": 10,
            "embedding_model": "BGE-large",
            "llm_model": "gpt-4o-mini",
        }
        mock_openai_module, _ = _make_mock_openai(new_config)

        history_path = tmp_path / "history.jsonl"
        history_path.write_text(
            json.dumps({"iteration": 1, "config": SAMPLE_CONFIG, "scores": {}, "composite_score": 0.75})
        )
        config_path = tmp_path / "experiment_config.json"
        notes_path = tmp_path / "agent_notes.md"

        with patch.dict(sys.modules, {"openai": mock_openai_module}):
            from src.agent import suggest_next_config

            result = suggest_next_config(
                history_path=history_path,
                search_space=SEARCH_SPACE,
                current_scores={"faithfulness": 0.8, "context_recall": 0.6},
                config_output_path=config_path,
                notes_path=notes_path,
            )

        assert result == new_config
        assert config_path.exists()
        assert json.loads(config_path.read_text()) == new_config
        assert notes_path.exists()
        notes = notes_path.read_text()
        assert "Iteration 2" in notes
        assert "Context recall is weakest" in notes

    def test_suggest_rejects_duplicate(self, tmp_path):
        new_config = {**SAMPLE_CONFIG, "chunk_size": 256}

        mock_openai_module = MagicMock()
        mock_client = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client

        # First call returns duplicate, second returns valid
        dup_response = MagicMock()
        dup_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "analysis": "...", "decision": "...", "config": SAMPLE_CONFIG,
            })))
        ]
        valid_response = MagicMock()
        valid_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "analysis": "...", "decision": "...", "config": new_config,
            })))
        ]
        mock_client.chat.completions.create.side_effect = [dup_response, valid_response]

        history_path = tmp_path / "history.jsonl"
        history_path.write_text(
            json.dumps({"iteration": 1, "config": SAMPLE_CONFIG, "composite_score": 0.75})
        )

        with patch.dict(sys.modules, {"openai": mock_openai_module}):
            from src.agent import suggest_next_config

            result = suggest_next_config(
                history_path=history_path,
                search_space=SEARCH_SPACE,
                current_scores={},
                config_output_path=tmp_path / "config.json",
                notes_path=tmp_path / "notes.md",
            )

        assert result == new_config
        assert mock_client.chat.completions.create.call_count == 2

    def test_suggest_fails_after_max_retries(self, tmp_path):
        invalid_config = {**SAMPLE_CONFIG, "chunk_size": 999}
        mock_openai_module, _ = _make_mock_openai(invalid_config)

        history_path = tmp_path / "history.jsonl"
        history_path.write_text("")

        with patch.dict(sys.modules, {"openai": mock_openai_module}):
            from src.agent import suggest_next_config

            with pytest.raises(RuntimeError, match="failed to suggest"):
                suggest_next_config(
                    history_path=history_path,
                    search_space=SEARCH_SPACE,
                    current_scores={},
                    config_output_path=tmp_path / "config.json",
                    notes_path=tmp_path / "notes.md",
                    max_retries=2,
                )
