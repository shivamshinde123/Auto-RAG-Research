"""Tests for git checkpoint (auto-commit on score improvement, file staging)."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.git_checkpoint import git_checkpoint, _is_git_available, _run_git


SAMPLE_CONFIG = {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "top_k": 5,
    "embedding_model": "all-MiniLM-L6-v2",
    "llm_model": "gpt-4o-mini",
}


class TestRunGit:
    def test_git_available(self):
        ok, _ = _run_git("--version")
        assert ok is True

    def test_git_invalid_command(self):
        ok, _ = _run_git("nonexistent-command")
        assert ok is False


class TestGitCheckpoint:
    def test_saves_best_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        # Create required experiment files
        (tmp_path / "experiment_history.jsonl").write_text("{}\n")
        (tmp_path / "experiment_config.json").write_text("{}\n")

        with patch("src.git_checkpoint._is_git_available", return_value=True):
            with patch("src.git_checkpoint._run_git", return_value=(True, "")):
                git_checkpoint(config=SAMPLE_CONFIG, score=0.85, run_number=3)

        best = tmp_path / "best_config.json"
        assert best.exists()
        assert json.loads(best.read_text()) == SAMPLE_CONFIG

    def test_commit_message_format(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "experiment_history.jsonl").write_text("{}\n")

        commits = []

        def mock_run_git(*args):
            if args[0] == "commit":
                commits.append(args)
            return True, ""

        with patch("src.git_checkpoint._is_git_available", return_value=True):
            with patch("src.git_checkpoint._run_git", side_effect=mock_run_git):
                git_checkpoint(config=SAMPLE_CONFIG, score=0.8512, run_number=5)

        assert len(commits) == 1
        msg = commits[0][2]  # -m, message
        assert "experiment 5" in msg
        assert "composite_score=0.8512" in msg
        assert "chunk_size=512" in msg

    def test_disabled_skips(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("src.git_checkpoint._run_git") as mock:
            git_checkpoint(config=SAMPLE_CONFIG, score=0.8, run_number=1, enabled=False)
            mock.assert_not_called()

    def test_no_git_repo_skips(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("src.git_checkpoint._is_git_available", return_value=False):
            with patch("src.git_checkpoint._run_git") as mock:
                git_checkpoint(config=SAMPLE_CONFIG, score=0.8, run_number=1)
                # Only the _is_git_available call, no git add/commit
                mock.assert_not_called()

    def test_stages_existing_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "experiment_history.jsonl").write_text("{}\n")
        (tmp_path / "agent_notes.md").write_text("notes\n")
        (tmp_path / "experiment_config.json").write_text("{}\n")

        staged_files = []

        def mock_run_git(*args):
            if args[0] == "add":
                staged_files.append(args[1])
            return True, ""

        with patch("src.git_checkpoint._is_git_available", return_value=True):
            with patch("src.git_checkpoint._run_git", side_effect=mock_run_git):
                git_checkpoint(config=SAMPLE_CONFIG, score=0.8, run_number=1)

        # best_config.json is created, plus the 3 existing files
        assert "best_config.json" in staged_files
        assert "experiment_history.jsonl" in staged_files
        assert "agent_notes.md" in staged_files
        assert "experiment_config.json" in staged_files
