"""AutoRAGResearch — Autonomous RAG pipeline optimization system.

Runs a closed-loop experiment cycle:
config -> RAG pipeline -> RAGAS evaluation -> LLM agent -> repeat.
"""

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.config_loader import load_config
from src.cost_tracker import CostTracker
from src.data_sources import get_data_source

logger = logging.getLogger("autorag")


def _setup_logging(verbose: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _random_config(search_space) -> dict:
    """Pick a random config from the search space."""
    return {
        "chunk_size": random.choice(search_space.chunk_size),
        "chunk_overlap": random.choice(search_space.chunk_overlap),
        "top_k": random.choice(search_space.top_k),
        "embedding_model": random.choice(search_space.embedding_model),
        "llm_model": random.choice(search_space.llm_model),
    }


def _search_space_as_dict(search_space) -> dict:
    """Convert SearchSpace dataclass to a plain dict for the agent."""
    return {
        "chunk_size": search_space.chunk_size,
        "chunk_overlap": search_space.chunk_overlap,
        "top_k": search_space.top_k,
        "embedding_model": search_space.embedding_model,
        "llm_model": search_space.llm_model,
    }


def _build_data_source_configs(program_config) -> list[dict]:
    """Convert DataSourceConfig dataclasses to plain dicts for dataset_loader."""
    configs = []
    for ds in program_config.data_sources:
        config = {"type": ds.type, "enabled": ds.enabled, **ds.extras}
        configs.append(config)
    return configs


def _load_existing_history(history_path: Path) -> list[dict]:
    """Load existing experiment history for resume support."""
    if not history_path.exists():
        return []
    entries = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def _validate_credentials(program_config) -> bool:
    """Validate credentials for all enabled data sources at startup.

    Reports all issues at once rather than failing on first error.
    Returns True if all validations pass, False otherwise.
    """
    ds_configs = _build_data_source_configs(program_config)
    enabled = [c for c in ds_configs if c.get("enabled", False)]

    if not enabled:
        logger.warning("No enabled data sources")
        return True

    issues = []
    for config in enabled:
        source_type = config.get("type", "unknown")
        try:
            source = get_data_source(config)
            source.validate_config()
            logger.info("  [OK] %s", source_type)
        except Exception as e:
            issues.append(f"  [FAIL] {source_type}: {e}")
            logger.error("  [FAIL] %s: %s", source_type, e)

    if issues:
        logger.error("Credential validation failed:\n%s", "\n".join(issues))
        return False

    return True


def run_experiment(config_path: str, dry_run: bool = False, resume: bool = False):
    """Run the full experiment loop.

    Args:
        config_path: Path to program.md config file.
        dry_run: If True, validate config and exit.
        resume: If True, continue from existing experiment_history.jsonl.
    """
    from src.dataset_loader import load_documents
    from src.evaluator import evaluate
    from src.experiment_logger import ExperimentLogger
    from src.git_checkpoint import git_checkpoint
    from src.rag_pipeline import run_pipeline

    # Load and validate config
    program_config = load_config(config_path)
    logger.info("Loaded config from %s", config_path)
    logger.info(
        "Search space: %s", json.dumps(_search_space_as_dict(program_config.search_space))
    )

    # Validate credentials for all enabled sources
    logger.info("Validating data source credentials...")
    if not _validate_credentials(program_config):
        logger.error("Fix credential issues above and retry")
        sys.exit(1)

    if dry_run:
        logger.info("Dry run — config and credentials valid. Exiting.")
        print("Config validation passed.")
        return

    # Set up paths
    history_path = Path("experiment_history.jsonl")
    config_output_path = Path("experiment_config.json")
    notes_path = Path("agent_notes.md")
    best_config_path = Path("best_config.json")

    # Load documents
    ds_configs = _build_data_source_configs(program_config)
    documents, qa_pairs = load_documents(ds_configs)

    if not documents:
        logger.error("No documents loaded — cannot run pipeline")
        sys.exit(1)

    if not qa_pairs:
        logger.error("No QA pairs available — cannot evaluate. Enable a huggingface data source.")
        sys.exit(1)

    # Initialize experiment logger and cost tracker
    exp_logger = ExperimentLogger(program_config.experiment.experiment_name)
    cost_tracker = CostTracker(max_cost_usd=program_config.constraints.max_cost_usd)

    # Resume or start fresh
    existing_history = _load_existing_history(history_path) if resume else []
    start_iteration = len(existing_history) + 1
    best_score = -1.0
    best_config = None

    # Find best from history if resuming
    for entry in existing_history:
        if entry.get("composite_score", -1) > best_score:
            best_score = entry["composite_score"]
            best_config = entry.get("config")

    max_iterations = program_config.constraints.max_iterations
    min_threshold = program_config.optimization.min_threshold
    search_space_dict = _search_space_as_dict(program_config.search_space)

    # For iteration 1 (or resume start): pick random config
    if start_iteration == 1 or not config_output_path.exists():
        current_config = _random_config(program_config.search_space)
        config_output_path.write_text(
            json.dumps(current_config, indent=2) + "\n", encoding="utf-8"
        )

    logger.info("Starting experiment loop (iterations %d-%d)", start_iteration, max_iterations)

    try:
        for iteration in range(start_iteration, max_iterations + 1):
            logger.info("=" * 60)
            logger.info("Iteration %d / %d", iteration, max_iterations)

            # Read current config
            current_config = json.loads(
                config_output_path.read_text(encoding="utf-8")
            )
            logger.info("Config: %s", json.dumps(current_config))

            # Run RAG pipeline
            results = run_pipeline(documents, qa_pairs, current_config)

            if not results:
                logger.warning("Pipeline returned no results, skipping evaluation")
                continue

            # Evaluate
            scores = evaluate(results)
            composite = scores.get("composite_score", -1)
            logger.info("Scores: composite=%.4f", composite)

            # Append to history
            history_entry = {
                "iteration": iteration,
                "config": current_config,
                "scores": scores,
                "composite_score": composite,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(history_entry) + "\n")

            # Track iteration cost
            iteration_cost = cost_tracker.end_iteration()

            # Log to MLflow
            is_best = composite > best_score
            scores_with_cost = {**scores, "iteration_cost_usd": iteration_cost, "total_cost_usd": cost_tracker.total_cost}
            exp_logger.log_run(
                config=current_config,
                scores=scores_with_cost,
                run_number=iteration,
                is_best=is_best,
            )

            # Check for improvement
            if is_best:
                best_score = composite
                best_config = current_config
                logger.info("New best score: %.4f", best_score)

                # Git checkpoint
                if program_config.experiment.git_checkpoints:
                    git_checkpoint(
                        config=current_config,
                        score=composite,
                        run_number=iteration,
                    )

            # Check stopping conditions
            if composite >= min_threshold:
                logger.info(
                    "Target reached! composite=%.4f >= threshold=%.4f",
                    composite, min_threshold,
                )
                break

            if cost_tracker.budget_exceeded():
                logger.info(
                    "Budget exceeded: $%.4f >= $%.2f",
                    cost_tracker.total_cost, cost_tracker.max_cost_usd,
                )
                break

            if iteration >= max_iterations:
                logger.info("Max iterations reached (%d)", max_iterations)
                break

            # Call agent for next config
            from src.agent import suggest_next_config

            suggest_next_config(
                history_path=history_path,
                search_space=search_space_dict,
                current_scores=scores,
                config_output_path=config_output_path,
                notes_path=notes_path,
            )

    except KeyboardInterrupt:
        logger.info("Interrupted by user — saving state")

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    if best_config:
        print(f"Best composite score: {best_score:.4f}")
        print(f"Best config: {json.dumps(best_config, indent=2)}")
        # Save best config
        best_config_path.write_text(
            json.dumps(best_config, indent=2) + "\n", encoding="utf-8"
        )
    else:
        print("No successful runs completed.")
    cost_summary = cost_tracker.summary()
    print(f"Total cost: ${cost_summary['total_cost_usd']:.4f} / ${cost_summary['max_cost_usd']:.2f}")
    print("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AutoRAGResearch — Autonomous RAG pipeline optimization"
    )
    parser.add_argument(
        "--config",
        default="program.md",
        help="Path to program.md config file (default: program.md)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without running experiments",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from existing experiment_history.jsonl",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging for detailed output",
    )
    args = parser.parse_args()

    _setup_logging(verbose=args.verbose)
    run_experiment(args.config, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
