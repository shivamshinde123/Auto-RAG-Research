"""Git checkpoint for auto-committing when composite score improves.

Saves the best config to best_config.json and creates a git commit
with experiment metadata in the commit message.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

# Files to stage when creating a git checkpoint commit
EXPERIMENT_FILES = [
    "best_config.json",
    "experiment_history.jsonl",
    "agent_notes.md",
    "experiment_config.json",
]


def _run_git(*args: str) -> Tuple[bool, str]:
    """Run a git command and return (success, output)."""
    try:
        result = subprocess.run(
            ["git"] + list(args),
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0, result.stdout.strip()
    except FileNotFoundError:
        return False, "git not found"
    except subprocess.TimeoutExpired:
        return False, "git command timed out"


def _is_git_available() -> bool:
    """Check if we're in a git repo."""
    ok, _ = _run_git("rev-parse", "--is-inside-work-tree")
    return ok


def git_checkpoint(
    config: dict,
    score: float,
    run_number: int,
    enabled: bool = True,
):
    """Save best config and create a git commit.

    Args:
        config: The best config dict.
        score: The composite score.
        run_number: The experiment iteration number.
        enabled: Whether git checkpoints are enabled.
    """
    if not enabled:
        logger.debug("Git checkpoints disabled, skipping")
        return

    if not _is_git_available():
        logger.warning("Not in a git repository, skipping checkpoint")
        return

    # Save best_config.json
    Path("best_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    # Stage experiment files
    files_to_stage = [f for f in EXPERIMENT_FILES if Path(f).exists()]
    if not files_to_stage:
        logger.warning("No experiment files to stage")
        return

    staging_failed = False
    for f in files_to_stage:
        ok, output = _run_git("add", f)
        if not ok:
            logger.error("Failed to stage %s: %s", f, output)
            staging_failed = True

    if staging_failed:
        logger.error("Aborting git checkpoint — one or more files failed to stage")
        return

    # Build commit message
    msg = (
        f"experiment {run_number}: composite_score={score:.4f} | "
        f"chunk_size={config.get('chunk_size', '?')}, "
        f"chunk_overlap={config.get('chunk_overlap', '?')}, "
        f"top_k={config.get('top_k', '?')}, "
        f"embedding={config.get('embedding_model', '?')}, "
        f"llm={config.get('llm_model', '?')}"
    )

    ok, output = _run_git("commit", "-m", msg)
    if ok:
        logger.info("Git checkpoint: %s", msg)
    else:
        if "nothing to commit" in output:
            logger.info("No changes to commit for checkpoint")
        else:
            logger.warning("Git commit failed: %s", output)
