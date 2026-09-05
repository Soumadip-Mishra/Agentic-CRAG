"""
Unit tests for the CRAG agent state machine routing.
"""

from __future__ import annotations

import pytest

from src.agent.edges import route_after_grading, route_after_hallucination_check
from src.agent.state import GraphState


class TestRouteAfterGrading:
    """Tests for the post-grading routing logic."""

    def _make_state(self, decision: str, retry_count: int = 0) -> GraphState:
        return {
            "question": "test question",
            "rewritten_question": "",
            "documents": [],
            "web_results": [],
            "relevance_scores": [],
            "relevance_decision": decision,
            "generation": "",
            "hallucination_score": 0.0,
            "retry_count": retry_count,
            "route": "",
        }

    def test_relevant_routes_to_generate(self):
        state = self._make_state("relevant")
        assert route_after_grading(state) == "generate"

    def test_ambiguous_routes_to_rewrite(self):
        state = self._make_state("ambiguous", retry_count=0)
        assert route_after_grading(state) == "rewrite"

    def test_irrelevant_routes_to_web_search(self):
        state = self._make_state("irrelevant")
        assert route_after_grading(state) == "web_search"

    def test_ambiguous_with_max_retries_routes_to_web_search(self):
        """When retries are exhausted, ambiguous should fall back to web search."""
        state = self._make_state("ambiguous", retry_count=10)
        assert route_after_grading(state) == "web_search"


class TestRouteAfterHallucinationCheck:
    """Tests for post-hallucination-check routing logic."""

    def _make_state(self, score: float, retry_count: int = 0) -> GraphState:
        return {
            "question": "test question",
            "rewritten_question": "",
            "documents": [],
            "web_results": [],
            "relevance_scores": [],
            "relevance_decision": "relevant",
            "generation": "test answer",
            "hallucination_score": score,
            "retry_count": retry_count,
            "route": "",
        }

    def test_high_score_routes_to_end(self):
        state = self._make_state(0.95)
        assert route_after_hallucination_check(state) == "end"

    def test_low_score_routes_to_rewrite(self):
        state = self._make_state(0.2, retry_count=0)
        assert route_after_hallucination_check(state) == "rewrite"

    def test_low_score_with_max_retries_routes_to_end(self):
        state = self._make_state(0.2, retry_count=10)
        assert route_after_hallucination_check(state) == "end"
