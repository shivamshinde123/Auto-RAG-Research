"""Evaluator using RAGAS metrics and composite scoring."""

import logging
import math
from typing import List

logger = logging.getLogger(__name__)

METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def evaluate(
    results: List[dict],
    judge_model: str = "openai",
) -> dict:
    """Run RAGAS evaluation on pipeline results.

    Args:
        results: List of dicts with question, answer, contexts, ground_truth.
        judge_model: "openai" (default) or "ollama" for local evaluation.

    Returns:
        Dict with faithfulness, answer_relevancy, context_precision,
        context_recall, and composite_score. Failed metrics return NaN.
    """
    from datasets import Dataset  # noqa: F811
    from ragas import evaluate as ragas_evaluate  # noqa: F811
    from ragas.metrics import (  # noqa: F811
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    # Build RAGAS dataset
    ragas_data = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    }
    dataset = Dataset.from_dict(ragas_data)

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    # Configure judge LLM
    llm = None
    embeddings = None
    if judge_model == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama
            from langchain_community.embeddings import OllamaEmbeddings
            llm = ChatOllama(model="llama3")
            embeddings = OllamaEmbeddings(model="llama3")
        except Exception as e:
            logger.warning("Failed to set up Ollama judge, falling back to OpenAI: %s", e)

    try:
        kwargs = {"dataset": dataset, "metrics": metrics}
        if llm is not None:
            kwargs["llm"] = llm
        if embeddings is not None:
            kwargs["embeddings"] = embeddings

        ragas_result = ragas_evaluate(**kwargs)
        scores = {}
        for name in METRIC_NAMES:
            val = ragas_result.get(name, float("nan"))
            scores[name] = val if val is not None else float("nan")
    except Exception as e:
        logger.error("RAGAS evaluation failed: %s", e)
        scores = {name: float("nan") for name in METRIC_NAMES}

    scores["composite_score"] = compute_composite(scores)

    logger.info(
        "Evaluation complete: composite=%.4f | %s",
        scores["composite_score"],
        " | ".join(f"{k}={v:.4f}" for k, v in scores.items() if k != "composite_score"),
    )
    return scores


def compute_composite(
    scores: dict,
    weights: dict | None = None,
) -> float:
    """Compute weighted average of metric scores, ignoring NaN values.

    Args:
        scores: Dict with metric names as keys and float scores as values.
        weights: Optional dict mapping metric names to weights. Defaults to equal.

    Returns:
        Weighted average score, or NaN if all metrics are NaN.
    """
    if weights is None:
        weights = {name: 1.0 for name in METRIC_NAMES}

    total_weight = 0.0
    weighted_sum = 0.0

    for name in METRIC_NAMES:
        val = scores.get(name, float("nan"))
        w = weights.get(name, 1.0)
        if not math.isnan(val):
            weighted_sum += val * w
            total_weight += w

    if total_weight == 0:
        return float("nan")

    return weighted_sum / total_weight
