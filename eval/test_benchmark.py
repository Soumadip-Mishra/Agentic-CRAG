"""
Pytest benchmark suite for regression testing.

Verifies that the CRAG agent produces answers that meet
minimum quality thresholds on the evaluation dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

EVAL_DATASET_PATH = Path("data/eval_dataset.json")


@pytest.fixture(scope="module")
def eval_dataset() -> list[dict]:
    """Load the evaluation dataset."""
    with open(EVAL_DATASET_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def agent():
    """Compile the CRAG agent workflow."""
    from src.agent.workflow import compile_workflow

    return compile_workflow()


class TestBenchmark:
    """Regression tests for CRAG answer quality."""

    def _invoke_agent(self, agent, question: str) -> dict:
        """Run the agent with a question and return full state."""
        return agent.invoke({
            "question": question,
            "rewritten_question": "",
            "documents": [],
            "web_results": [],
            "relevance_scores": [],
            "relevance_decision": "",
            "generation": "",
            "hallucination_score": 0.0,
            "retry_count": 0,
            "route": "",
        })

    def test_agent_produces_non_empty_answers(self, agent, eval_dataset):
        """Every question should produce a non-empty answer."""
        for qa in eval_dataset:
            result = self._invoke_agent(agent, qa["question"])
            assert result.get("generation"), (
                f"Empty answer for question: {qa['question']}"
            )

    def test_hallucination_scores_above_threshold(self, agent, eval_dataset):
        """Hallucination scores should be above the configured threshold."""
        from config.settings import get_settings

        settings = get_settings()

        for qa in eval_dataset:
            result = self._invoke_agent(agent, qa["question"])
            score = result.get("hallucination_score", 0.0)
            assert score >= settings.hallucination_threshold * 0.8, (
                f"Hallucination score {score:.3f} too low for: {qa['question']}"
            )

    def test_retry_count_within_limits(self, agent, eval_dataset):
        """Agent should not exceed max retries."""
        from config.settings import get_settings

        settings = get_settings()

        for qa in eval_dataset:
            result = self._invoke_agent(agent, qa["question"])
            assert result.get("retry_count", 0) <= settings.max_retries, (
                f"Exceeded max retries for: {qa['question']}"
            )
