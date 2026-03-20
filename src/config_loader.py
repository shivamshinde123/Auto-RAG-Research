"""Parses program.md config file into structured Python objects."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """Config for a single data source (e.g., local_pdf).

    Known fields (type, enabled) are stored directly; any extra
    source-specific fields (path, etc.) go into the extras dict.
    """

    type: str
    enabled: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a source-specific config value from extras."""
        return self.extras.get(key, default)


@dataclass
class SearchSpace:
    """Hyperparameter search space — each field is a list of allowed values."""

    chunk_size: List[int] = field(default_factory=lambda: [512])
    chunk_overlap: List[int] = field(default_factory=lambda: [50])
    top_k: List[int] = field(default_factory=lambda: [5])
    embedding_model: List[str] = field(default_factory=lambda: ["all-MiniLM-L6-v2"])
    llm_model: List[str] = field(default_factory=lambda: ["gpt-4o-mini"])


@dataclass
class OptimizationTarget:
    """Which RAGAS metric to optimize and when to stop."""

    primary_metric: str = "context_recall"
    secondary_metric: str = "faithfulness"
    min_threshold: float = 0.80


@dataclass
class Constraints:
    """Budget and iteration limits for the experiment loop."""

    max_iterations: int = 20
    max_cost_usd: float = 5.0


@dataclass
class ExperimentConfig:
    """Experiment metadata — name and git checkpoint toggle."""

    experiment_name: str = "autoragresearch_run_1"
    git_checkpoints: bool = True


@dataclass
class QAGenerationConfig:
    """How many QA pairs to auto-generate from PDF content."""

    num_qa_pairs: int = 20


@dataclass
class ProgramConfig:
    """Top-level config container — aggregates all sections from program.md."""

    data_sources: List[DataSourceConfig] = field(default_factory=list)
    search_space: SearchSpace = field(default_factory=SearchSpace)
    optimization: OptimizationTarget = field(default_factory=OptimizationTarget)
    constraints: Constraints = field(default_factory=Constraints)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    qa_generation: QAGenerationConfig = field(default_factory=QAGenerationConfig)


def _parse_value(raw: str) -> Any:
    """Parse a raw string value into the appropriate Python type."""
    raw = raw.strip()

    # Try types in order: bool -> int -> float -> list -> string
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False

    try:
        return int(raw)
    except ValueError:
        pass

    try:
        return float(raw)
    except ValueError:
        pass

    # Bracket-delimited list: [item1, item2, item3]
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1]
        items = [item.strip().strip("'\"") for item in inner.split(",")]
        # Attempt numeric conversion for each list item
        parsed_items = []
        for item in items:
            try:
                parsed_items.append(int(item))
            except ValueError:
                try:
                    parsed_items.append(float(item))
                except ValueError:
                    parsed_items.append(item)
        return parsed_items

    # Fall through: treat as plain string
    return raw


def _parse_lines(lines: List[str]) -> ProgramConfig:
    """Parse the lines of a program.md file into a ProgramConfig."""
    config = ProgramConfig()
    current_section: Optional[str] = None
    current_data_source: Optional[Dict[str, Any]] = None
    multiline_key: Optional[str] = None
    multiline_list: Optional[List[str]] = None

    # State machine: walk lines, switching on current section header
    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        # Match markdown H2 headers (## Section Name) to determine current section
        section_match = re.match(r"^##\s+(.+)$", stripped)
        if section_match:
            # Flush any in-progress multiline list
            if multiline_key and multiline_list is not None and current_data_source is not None:
                current_data_source[multiline_key] = multiline_list
                multiline_key = None
                multiline_list = None

            # Flush any in-progress data source
            if current_data_source is not None:
                config.data_sources.append(_build_data_source(current_data_source))
                current_data_source = None

            current_section = section_match.group(1).strip().lower()
            continue

        # Skip comment lines (after section header check, since ## is also a comment prefix)
        if stripped.startswith("#"):
            continue

        # Data source block start: [[data_sources]]
        if stripped == "[[data_sources]]":
            # Flush any in-progress multiline list
            if multiline_key and multiline_list is not None and current_data_source is not None:
                current_data_source[multiline_key] = multiline_list
                multiline_key = None
                multiline_list = None

            # Flush previous data source
            if current_data_source is not None:
                config.data_sources.append(_build_data_source(current_data_source))

            current_data_source = {}
            current_section = "data_sources"
            continue

        # Multiline list item:   - value
        if multiline_key is not None and multiline_list is not None:
            list_item_match = re.match(r"^\s+-\s+(.+)$", line)
            if list_item_match:
                multiline_list.append(list_item_match.group(1).strip())
                continue
            else:
                # End of multiline list
                if current_data_source is not None:
                    current_data_source[multiline_key] = multiline_list
                multiline_key = None
                multiline_list = None
                # Fall through to parse current line as key:value

        # Key-value pair: key: value
        kv_match = re.match(r"^(\w+):\s*$", stripped)
        if kv_match:
            # Key with no value — start of a multiline list
            multiline_key = kv_match.group(1)
            multiline_list = []
            continue

        kv_match = re.match(r"^(\w+):\s+(.+)$", stripped)
        if kv_match:
            key = kv_match.group(1)
            value = _parse_value(kv_match.group(2))

            if current_section == "data_sources" and current_data_source is not None:
                current_data_source[key] = value
            elif current_section == "search space":
                if hasattr(config.search_space, key):
                    setattr(config.search_space, key, value if isinstance(value, list) else [value])
                else:
                    logger.warning("Unrecognized search space key '%s' — ignoring", key)
            elif current_section == "optimization target":
                if hasattr(config.optimization, key):
                    setattr(config.optimization, key, value)
                else:
                    logger.warning("Unrecognized optimization key '%s' — ignoring", key)
            elif current_section == "constraints":
                if hasattr(config.constraints, key):
                    setattr(config.constraints, key, value)
                else:
                    logger.warning("Unrecognized constraints key '%s' — ignoring", key)
            elif current_section == "experiment":
                if hasattr(config.experiment, key):
                    setattr(config.experiment, key, value)
                else:
                    logger.warning("Unrecognized experiment key '%s' — ignoring", key)
            elif current_section == "qa generation":
                if hasattr(config.qa_generation, key):
                    setattr(config.qa_generation, key, value)
                else:
                    logger.warning("Unrecognized qa generation key '%s' — ignoring", key)
            continue

    # Flush final multiline list
    if multiline_key and multiline_list is not None and current_data_source is not None:
        current_data_source[multiline_key] = multiline_list
        multiline_key = None
        multiline_list = None

    # Flush final data source
    if current_data_source is not None:
        config.data_sources.append(_build_data_source(current_data_source))

    return config


def _build_data_source(raw: Dict[str, Any]) -> DataSourceConfig:
    """Build a DataSourceConfig from a raw dict."""
    ds_type = raw.pop("type", None)
    if ds_type is None:
        raise ValueError(
            "Data source missing required 'type' field. "
            "Each [[data_sources]] block must include 'type: <name>' "
            "(e.g., type: local_pdf, type: huggingface)."
        )

    enabled = raw.pop("enabled", False)
    return DataSourceConfig(type=ds_type, enabled=enabled, extras=raw)


def load_config(path: Union[str, Path]) -> ProgramConfig:
    """Load and parse a program.md config file.

    Args:
        path: Path to the program.md file.

    Returns:
        A ProgramConfig dataclass with all parsed settings.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file has invalid content.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            "Create a program.md config file or pass a valid path with --config."
        )

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    config = _parse_lines(lines)
    _validate_config(config)
    return config


def _validate_config(config: ProgramConfig) -> None:
    """Validate the parsed config for required fields and sane values."""
    # Must have at least one data source
    if not config.data_sources:
        raise ValueError(
            "Config must define at least one [[data_sources]] block. "
            "Add a [[data_sources]] section to your program.md file."
        )

    # Every data source must have a type
    for i, ds in enumerate(config.data_sources):
        if not ds.type:
            raise ValueError(f"Data source #{i + 1} missing required 'type' field")

    # Search space must have non-empty lists
    for field_name in ("chunk_size", "chunk_overlap", "top_k", "embedding_model", "llm_model"):
        values = getattr(config.search_space, field_name)
        if not values:
            raise ValueError(f"Search space '{field_name}' must have at least one value")

    # Numeric constraints must be positive
    if config.constraints.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if config.constraints.max_cost_usd <= 0:
        raise ValueError("max_cost_usd must be positive")

    # Threshold must be between 0 and 1
    if not (0.0 <= config.optimization.min_threshold <= 1.0):
        raise ValueError("min_threshold must be between 0.0 and 1.0")

    # Primary metric must be valid
    valid_metrics = {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
    if config.optimization.primary_metric not in valid_metrics:
        raise ValueError(
            f"primary_metric must be one of {valid_metrics}, "
            f"got '{config.optimization.primary_metric}'"
        )
    if config.optimization.secondary_metric not in valid_metrics:
        raise ValueError(
            f"secondary_metric must be one of {valid_metrics}, "
            f"got '{config.optimization.secondary_metric}'"
        )
