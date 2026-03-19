"""LLM agent for config suggestion with reasoning chain.

The agent reads experiment history and search space constraints,
then writes a new experiment_config.json and appends reasoning to agent_notes.md.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_history(history_path: Path) -> list[dict]:
    """Load experiment history from JSONL file."""
    if not history_path.exists():
        return []
    entries = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def _build_prompt(search_space: dict, history: list[dict], current_scores: dict) -> str:
    """Build the agent prompt for suggesting the next config."""
    history_text = ""
    if history:
        for entry in history:
            cfg = entry.get("config", {})
            scores = entry.get("scores", {})
            history_text += (
                f"  Iteration {entry.get('iteration', '?')}: "
                f"composite={entry.get('composite_score', '?'):.4f} | "
                f"config={json.dumps(cfg)} | "
                f"scores={json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in scores.items()})}\n"
            )
    else:
        history_text = "  (no previous experiments)\n"

    tried_configs = [json.dumps(e.get("config", {}), sort_keys=True) for e in history]
    tried_text = "\n".join(f"  - {c}" for c in tried_configs) if tried_configs else "  (none)"

    return f"""You are an AI research agent optimizing a RAG pipeline. Analyze the experiment history and suggest the next configuration to try.

## Search Space Constraints
{json.dumps(search_space, indent=2)}

## Experiment History
{history_text}

## Current Scores
{json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in current_scores.items()}, indent=2)}

## Already Tried Configs (DO NOT repeat these)
{tried_text}

## Instructions
1. Analyze which metric is weakest and why
2. Reason about which hyperparameters would improve the weakest metric
3. Suggest a NEW config that has NOT been tried before
4. All values must be within the search space constraints

Respond in this exact JSON format:
{{
  "analysis": "Your analysis of the weakest metric and why...",
  "decision": "Your reasoning for the next config choice...",
  "config": {{
    "chunk_size": <int>,
    "chunk_overlap": <int>,
    "top_k": <int>,
    "embedding_model": "<string>",
    "llm_model": "<string>"
  }}
}}"""


def _validate_config(config: dict, search_space: dict) -> bool:
    """Validate that config values are within the search space."""
    for key, allowed in search_space.items():
        if key in config:
            if config[key] not in allowed:
                raise ValueError(
                    f"Config {key}={config[key]} not in allowed values: {allowed}"
                )
    return True


def _is_duplicate(config: dict, history: list[dict]) -> bool:
    """Check if a config has already been tried."""
    config_str = json.dumps(config, sort_keys=True)
    for entry in history:
        if json.dumps(entry.get("config", {}), sort_keys=True) == config_str:
            return True
    return False


def suggest_next_config(
    history_path: str | Path,
    search_space: dict,
    current_scores: dict,
    config_output_path: str | Path = "experiment_config.json",
    notes_path: str | Path = "agent_notes.md",
    llm_model: str = "gpt-4o-mini",
    max_retries: int = 3,
) -> dict:
    """Use an LLM to suggest the next experiment config.

    Args:
        history_path: Path to experiment_history.jsonl.
        search_space: Dict mapping param names to lists of allowed values.
        current_scores: Current RAGAS scores dict.
        config_output_path: Where to write experiment_config.json.
        notes_path: Where to append agent reasoning (agent_notes.md).
        llm_model: LLM to use for the agent.
        max_retries: Max attempts if agent suggests duplicate/invalid config.

    Returns:
        The suggested config dict.
    """
    from openai import OpenAI

    history_path = Path(history_path)
    config_output_path = Path(config_output_path)
    notes_path = Path(notes_path)

    history = _load_history(history_path)
    iteration = len(history) + 1

    client = OpenAI()

    for attempt in range(max_retries):
        prompt = _build_prompt(search_space, history, current_scores)

        if attempt > 0:
            prompt += f"\n\nNOTE: Your previous suggestion was rejected (attempt {attempt + 1}/{max_retries}). Pick a DIFFERENT config."

        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        config = parsed["config"]
        analysis = parsed.get("analysis", "")
        decision = parsed.get("decision", "")

        # Validate
        try:
            _validate_config(config, search_space)
        except ValueError as e:
            logger.warning("Agent suggested invalid config (attempt %d): %s", attempt + 1, e)
            continue

        if _is_duplicate(config, history):
            logger.warning("Agent suggested duplicate config (attempt %d)", attempt + 1)
            continue

        # Write experiment_config.json
        config_output_path.write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        logger.info("Wrote next config to %s", config_output_path)

        # Append to agent_notes.md
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        notes_entry = (
            f"\n## Iteration {iteration} \u2014 {timestamp}\n"
            f"### Analysis\n{analysis}\n\n"
            f"### Decision\n{decision}\n\n"
            f"### Next Config\n"
            f"{', '.join(f'{k}={v}' for k, v in config.items())}\n"
        )
        with open(notes_path, "a", encoding="utf-8") as f:
            f.write(notes_entry)
        logger.info("Appended reasoning to %s", notes_path)

        return config

    raise RuntimeError(
        f"Agent failed to suggest a valid, non-duplicate config after {max_retries} attempts"
    )
