"""Cost tracking for API calls across the experiment loop."""

import logging
from typing import List

logger = logging.getLogger(__name__)

# Pricing per 1K tokens (as of 2024 — approximate)
MODEL_PRICING = {
    # LLM models (input / output per 1K tokens)
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    # OpenAI embedding models (per 1K tokens)
    "text-embedding-ada-002": {"input": 0.0001, "output": 0.0},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
    # Local embedding models (free)
    "BGE-large": {"input": 0.0, "output": 0.0},
    "bge-large": {"input": 0.0, "output": 0.0},
    "all-MiniLM-L6-v2": {"input": 0.0, "output": 0.0},
    "all-minilm-l6-v2": {"input": 0.0, "output": 0.0},
}


class CostTracker:
    """Tracks cumulative API costs across experiment iterations."""

    def __init__(self, max_cost_usd: float = 5.0):
        self.max_cost_usd = max_cost_usd
        self.total_cost = 0.0
        self.iteration_costs: List[float] = []
        self._current_iteration_cost = 0.0

    def add_cost(self, model: str, input_tokens: int, output_tokens: int = 0):
        """Add cost for an API call."""
        pricing = MODEL_PRICING.get(model)
        if pricing is None:
            logger.warning("Unknown model '%s' — using fallback pricing (may be inaccurate)", model)
            pricing = {"input": 0.001, "output": 0.002}
        cost = (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]
        self._current_iteration_cost += cost
        self.total_cost += cost

    def end_iteration(self) -> float:
        """End current iteration and return its cost."""
        cost = self._current_iteration_cost
        self.iteration_costs.append(cost)
        self._current_iteration_cost = 0.0
        logger.info("Iteration cost: $%.6f | Total: $%.6f / $%.2f", cost, self.total_cost, self.max_cost_usd)
        return cost

    def budget_exceeded(self) -> bool:
        """Check if total cost has exceeded the budget."""
        return self.total_cost >= self.max_cost_usd

    def remaining_budget(self) -> float:
        """Return remaining budget in USD."""
        return max(0.0, self.max_cost_usd - self.total_cost)

    def summary(self) -> dict:
        """Return cost summary."""
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "max_cost_usd": self.max_cost_usd,
            "iterations_tracked": len(self.iteration_costs),
            "avg_cost_per_iteration": round(
                self.total_cost / len(self.iteration_costs), 6
            ) if self.iteration_costs else 0.0,
        }
