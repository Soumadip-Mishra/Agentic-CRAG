"""
GraphState TypedDict definition.

Defines the shared state that flows through every node
in the LangGraph CRAG workflow.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """State shared across LangGraph nodes."""

    # ── Input ──
    question: str                        # Original user query
    rewritten_question: str              # Query after rewriting (if triggered)

    # ── Retrieval ──
    documents: list[Document]            # Retrieved & reranked documents
    web_results: list[Document]          # Web-search fallback results

    # ── Grading ──
    relevance_scores: list[float]        # Per-document relevance scores
    relevance_decision: str              # "relevant" | "ambiguous" | "irrelevant"

    # ── Generation ──
    generation: str                      # Final generated answer
    hallucination_score: float           # Hallucination guard score

    # ── Control Flow ──
    retry_count: int                     # Number of correction loops so far
    route: str                           # Next node to visit
