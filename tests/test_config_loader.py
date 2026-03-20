"""Tests for config_loader (program.md parsing, validation, type coercion)."""

import pytest
from pathlib import Path

from src.config_loader import load_config, _parse_value, ProgramConfig


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def valid_config_path(tmp_path):
    """Create a valid program.md config file with PDF data source."""
    config_text = """\
## Data Sources

[[data_sources]]
type: local_pdf
path: data/pdfs/
enabled: true

## QA Generation
num_qa_pairs: 15

## Search Space
chunk_size: [256, 512, 1024]
chunk_overlap: [0, 50, 100]
top_k: [3, 5, 8, 10]
embedding_model: [text-embedding-ada-002, BGE-large, all-MiniLM-L6-v2]
llm_model: [gpt-4o-mini, gpt-3.5-turbo]

## Optimization Target
primary_metric: context_recall
secondary_metric: faithfulness
min_threshold: 0.80

## Constraints
max_iterations: 20
max_cost_usd: 5.0

## Experiment
experiment_name: test_run_1
git_checkpoints: true
"""
    config_file = tmp_path / "program.md"
    config_file.write_text(config_text)
    return config_file


@pytest.fixture
def minimal_config_path(tmp_path):
    """Create a minimal valid config with just one PDF data source."""
    config_text = """\
## Data Sources

[[data_sources]]
type: local_pdf
path: data/pdfs/
enabled: true

## Search Space
chunk_size: [512]
chunk_overlap: [50]
top_k: [5]
embedding_model: [all-MiniLM-L6-v2]
llm_model: [gpt-4o-mini]

## Optimization Target
primary_metric: faithfulness
secondary_metric: context_recall
min_threshold: 0.70

## Constraints
max_iterations: 5
max_cost_usd: 1.0

## Experiment
experiment_name: minimal_test
git_checkpoints: false
"""
    config_file = tmp_path / "program.md"
    config_file.write_text(config_text)
    return config_file


class TestParseValue:
    def test_boolean_true(self):
        assert _parse_value("true") is True
        assert _parse_value("True") is True

    def test_boolean_false(self):
        assert _parse_value("false") is False
        assert _parse_value("False") is False

    def test_integer(self):
        assert _parse_value("42") == 42
        assert _parse_value("0") == 0

    def test_float(self):
        assert _parse_value("3.14") == 3.14
        assert _parse_value("0.80") == 0.80

    def test_list_of_ints(self):
        assert _parse_value("[256, 512, 1024]") == [256, 512, 1024]

    def test_list_of_strings(self):
        assert _parse_value("[gpt-4o-mini, gpt-3.5-turbo]") == ["gpt-4o-mini", "gpt-3.5-turbo"]

    def test_plain_string(self):
        assert _parse_value("context_recall") == "context_recall"

    def test_path_string(self):
        assert _parse_value("data/pdfs/") == "data/pdfs/"


class TestLoadConfig:
    def test_valid_config(self, valid_config_path):
        config = load_config(valid_config_path)

        assert isinstance(config, ProgramConfig)
        assert len(config.data_sources) == 1

    def test_data_sources_parsed(self, valid_config_path):
        config = load_config(valid_config_path)

        pdf_source = config.data_sources[0]
        assert pdf_source.type == "local_pdf"
        assert pdf_source.enabled is True
        assert pdf_source.get("path") == "data/pdfs/"

    def test_qa_generation_parsed(self, valid_config_path):
        config = load_config(valid_config_path)

        assert config.qa_generation.num_qa_pairs == 15

    def test_search_space_parsed(self, valid_config_path):
        config = load_config(valid_config_path)

        assert config.search_space.chunk_size == [256, 512, 1024]
        assert config.search_space.chunk_overlap == [0, 50, 100]
        assert config.search_space.top_k == [3, 5, 8, 10]
        assert config.search_space.embedding_model == [
            "text-embedding-ada-002", "BGE-large", "all-MiniLM-L6-v2"
        ]
        assert config.search_space.llm_model == ["gpt-4o-mini", "gpt-3.5-turbo"]

    def test_optimization_parsed(self, valid_config_path):
        config = load_config(valid_config_path)

        assert config.optimization.primary_metric == "context_recall"
        assert config.optimization.secondary_metric == "faithfulness"
        assert config.optimization.min_threshold == 0.80

    def test_constraints_parsed(self, valid_config_path):
        config = load_config(valid_config_path)

        assert config.constraints.max_iterations == 20
        assert config.constraints.max_cost_usd == 5.0

    def test_experiment_parsed(self, valid_config_path):
        config = load_config(valid_config_path)

        assert config.experiment.experiment_name == "test_run_1"
        assert config.experiment.git_checkpoints is True

    def test_minimal_config(self, minimal_config_path):
        config = load_config(minimal_config_path)

        assert len(config.data_sources) == 1
        assert config.data_sources[0].type == "local_pdf"
        assert config.constraints.max_iterations == 5
        assert config.experiment.git_checkpoints is False

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path/program.md")


class TestValidation:
    def test_no_data_sources(self, tmp_path):
        config_file = tmp_path / "program.md"
        config_file.write_text("""\
## Search Space
chunk_size: [512]
chunk_overlap: [50]
top_k: [5]
embedding_model: [all-MiniLM-L6-v2]
llm_model: [gpt-4o-mini]

## Optimization Target
primary_metric: faithfulness
secondary_metric: context_recall
min_threshold: 0.70

## Constraints
max_iterations: 5
max_cost_usd: 1.0

## Experiment
experiment_name: test
git_checkpoints: false
""")
        with pytest.raises(ValueError, match="at least one"):
            load_config(config_file)

    def test_negative_max_iterations(self, tmp_path):
        config_file = tmp_path / "program.md"
        config_file.write_text("""\
## Data Sources

[[data_sources]]
type: local_pdf
path: data/pdfs/
enabled: true

## Search Space
chunk_size: [512]
chunk_overlap: [50]
top_k: [5]
embedding_model: [all-MiniLM-L6-v2]
llm_model: [gpt-4o-mini]

## Optimization Target
primary_metric: faithfulness
secondary_metric: context_recall
min_threshold: 0.70

## Constraints
max_iterations: -1
max_cost_usd: 1.0

## Experiment
experiment_name: test
git_checkpoints: false
""")
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            load_config(config_file)

    def test_invalid_threshold(self, tmp_path):
        config_file = tmp_path / "program.md"
        config_file.write_text("""\
## Data Sources

[[data_sources]]
type: local_pdf
path: data/pdfs/
enabled: true

## Search Space
chunk_size: [512]
chunk_overlap: [50]
top_k: [5]
embedding_model: [all-MiniLM-L6-v2]
llm_model: [gpt-4o-mini]

## Optimization Target
primary_metric: faithfulness
secondary_metric: context_recall
min_threshold: 1.5

## Constraints
max_iterations: 5
max_cost_usd: 1.0

## Experiment
experiment_name: test
git_checkpoints: false
""")
        with pytest.raises(ValueError, match="min_threshold must be between"):
            load_config(config_file)

    def test_invalid_primary_metric(self, tmp_path):
        config_file = tmp_path / "program.md"
        config_file.write_text("""\
## Data Sources

[[data_sources]]
type: local_pdf
path: data/pdfs/
enabled: true

## Search Space
chunk_size: [512]
chunk_overlap: [50]
top_k: [5]
embedding_model: [all-MiniLM-L6-v2]
llm_model: [gpt-4o-mini]

## Optimization Target
primary_metric: invalid_metric
secondary_metric: context_recall
min_threshold: 0.70

## Constraints
max_iterations: 5
max_cost_usd: 1.0

## Experiment
experiment_name: test
git_checkpoints: false
""")
        with pytest.raises(ValueError, match="primary_metric must be one of"):
            load_config(config_file)
