"""Evaluator using RAGAS metrics and composite scoring."""

import logging
import math
from typing import List, Optional

logger = logging.getLogger(__name__)

# The four RAGAS metrics we track for every experiment iteration
METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

# RAGAS 0.4.x uses different internal column names for some metrics.
# This map translates our canonical names to the column names in the
# pandas DataFrame returned by ragas.evaluate().to_pandas().
_RAGAS_METRIC_NAME_MAP = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "context_precision": "llm_context_precision_with_reference",
    "context_recall": "context_recall",
}


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
    if not results:
        logger.warning("No results to evaluate — returning NaN scores")
        scores = {name: float("nan") for name in METRIC_NAMES}
        scores["composite_score"] = float("nan")
        return scores

    # RAGAS 0.4.x API: use EvaluationDataset with SingleTurnSample objects
    from ragas import EvaluationDataset, SingleTurnSample
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        _AnswerRelevancy,
        _Faithfulness,
        _LLMContextPrecisionWithReference,
        _LLMContextRecall,
    )

    # Convert pipeline results into RAGAS sample format
    samples = []
    for r in results:
        samples.append(SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        ))
    dataset = EvaluationDataset(samples=samples)

    # Note: metric classes use underscore prefix in RAGAS 0.4.x
    metrics = [
        _Faithfulness(),
        _AnswerRelevancy(),
        _LLMContextPrecisionWithReference(),
        _LLMContextRecall(),
    ]

    # Configure judge LLM and embeddings for RAGAS evaluation
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

    # Explicitly provide OpenAI embeddings — RAGAS's internal auto-init can fail
    # due to version mismatches between langchain-openai and ragas
    if embeddings is None:
        try:
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings()
        except Exception as e:
            logger.warning("Failed to create OpenAI embeddings: %s", e)

    try:
        kwargs = {"dataset": dataset, "metrics": metrics}
        if llm is not None:
            kwargs["llm"] = llm
        if embeddings is not None:
            kwargs["embeddings"] = embeddings

        ragas_result = ragas_evaluate(**kwargs)

        # RAGAS 0.4.x returns an EvaluationResult object (not a dict).
        # Convert to pandas DataFrame and compute mean per metric column.
        df = ragas_result.to_pandas()
        scores = {}
        for canonical_name in METRIC_NAMES:
            ragas_name = _RAGAS_METRIC_NAME_MAP[canonical_name]
            if ragas_name in df.columns:
                val = df[ragas_name].mean()
                scores[canonical_name] = val if val is not None else float("nan")
            else:
                scores[canonical_name] = float("nan")
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
    weights: Optional[dict] = None,
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
