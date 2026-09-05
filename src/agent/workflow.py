"""
LangGraph workflow builder & compilation.

Constructs the CRAG state machine:

    retrieve → grade → [generate | rewrite | web_search]
                              ↓
                     hallucination_guard → [end | rewrite]
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agent.edges import route_after_grading, route_after_hallucination_check
from src.agent.nodes import (
    generate,
    grade_documents,
    hallucination_guard,
    retrieve,
    rewrite_query,
    web_search,
)
from src.agent.state import GraphState


def build_workflow() -> StateGraph:
    """Construct the CRAG LangGraph workflow (uncompiled)."""
    workflow = StateGraph(GraphState)

    # ── Add Nodes ──
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    workflow.add_node("hallucination_guard", hallucination_guard)

    # ── Define Edges ──
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # After grading: branch
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "web_search": "web_search",
        },
    )

    # Rewrite loops back to retrieve
    workflow.add_edge("rewrite", "retrieve")

    # Web search goes to generate
    workflow.add_edge("web_search", "generate")

    # After generation: check hallucination
    workflow.add_edge("generate", "hallucination_guard")

    # After hallucination check: branch
    workflow.add_conditional_edges(
        "hallucination_guard",
        route_after_hallucination_check,
        {
            "end": END,
            "rewrite": "rewrite",
        },
    )

    return workflow


def compile_workflow():
    """Build and compile the CRAG workflow into a runnable graph."""
    workflow = build_workflow()
    return workflow.compile()


# Module-level convenience
app = compile_workflow()
