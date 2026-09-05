"""
Conditional routing functions for the CRAG LangGraph workflow.

Each function inspects GraphState and returns the name of the
next node to route to.
"""

from __future__ import annotations

from loguru import logger

from config.settings import get_settings
from src.agent.state import GraphState


def route_after_grading(state: GraphState) -> str:
    """
    Decide what to do after document grading.

    Routes:
        - "relevant"   → generate answer
        - "ambiguous"  → rewrite query (if retries remain)
        - "irrelevant" → web search fallback
    """
    decision = state.get("relevance_decision", "irrelevant")
    retry_count = state.get("retry_count", 0)
    settings = get_settings()

    if decision == "relevant":
        logger.info("[route] grading → generate")
        return "generate"

    if decision == "ambiguous" and retry_count < settings.max_retries:
        logger.info(f"[route] grading → rewrite (retry {retry_count + 1}/{settings.max_retries})")
        return "rewrite"

    logger.info("[route] grading → web_search (fallback)")
    return "web_search"


def route_after_hallucination_check(state: GraphState) -> str:
    """
    Decide what to do after hallucination grading.

    Routes:
        - score >= threshold → return answer (END)
        - score < threshold  → regenerate (if retries remain) or return with warning
    """
    score = state.get("hallucination_score", 0.0)
    retry_count = state.get("retry_count", 0)
    settings = get_settings()

    if score >= settings.hallucination_threshold:
        logger.info(f"[route] hallucination check passed (score={score:.3f}) → END")
        return "end"

    if retry_count < settings.max_retries:
        logger.info(f"[route] hallucination check failed (score={score:.3f}) → rewrite")
        return "rewrite"

    logger.warning(f"[route] hallucination check failed, max retries reached → END (with warning)")
    return "end"
