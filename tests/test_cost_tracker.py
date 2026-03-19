"""Tests for cost_tracker module."""

from src.cost_tracker import CostTracker


class TestCostTracker:
    def test_initial_state(self):
        tracker = CostTracker(max_cost_usd=5.0)
        assert tracker.total_cost == 0.0
        assert tracker.budget_exceeded() is False
        assert tracker.remaining_budget() == 5.0

    def test_add_cost(self):
        tracker = CostTracker(max_cost_usd=5.0)
        tracker.add_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        assert tracker.total_cost > 0

    def test_budget_exceeded(self):
        tracker = CostTracker(max_cost_usd=0.001)
        tracker.add_cost("gpt-4o-mini", input_tokens=100000, output_tokens=100000)
        assert tracker.budget_exceeded() is True

    def test_end_iteration(self):
        tracker = CostTracker()
        tracker.add_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        cost = tracker.end_iteration()
        assert cost > 0
        assert len(tracker.iteration_costs) == 1
        # New iteration starts at 0
        tracker.add_cost("gpt-4o-mini", input_tokens=500, output_tokens=250)
        cost2 = tracker.end_iteration()
        assert cost2 < cost  # fewer tokens
        assert len(tracker.iteration_costs) == 2

    def test_summary(self):
        tracker = CostTracker(max_cost_usd=10.0)
        tracker.add_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        tracker.end_iteration()
        summary = tracker.summary()
        assert summary["max_cost_usd"] == 10.0
        assert summary["iterations_tracked"] == 1
        assert summary["total_cost_usd"] > 0

    def test_unknown_model_uses_fallback_pricing(self):
        tracker = CostTracker()
        tracker.add_cost("some-unknown-model", input_tokens=1000, output_tokens=500)
        assert tracker.total_cost > 0

    def test_remaining_budget(self):
        tracker = CostTracker(max_cost_usd=1.0)
        tracker.add_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        remaining = tracker.remaining_budget()
        assert remaining < 1.0
        assert remaining > 0
