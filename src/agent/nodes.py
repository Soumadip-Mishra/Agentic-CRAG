"""
Node implementations for the CRAG LangGraph workflow.

Each function takes a GraphState dict and returns
partial state updates.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from config.settings import get_settings
from src.agent.state import GraphState


def _parse_score(raw_text: str, default: float = 0.0) -> float:
    """Safely extract a float score even if wrapped in markdown code fences."""
    try:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
        data = json.loads(cleaned)
        return float(data.get("score", default))
    except Exception:
        match = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', raw_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return default


from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.sparse import SparseRetriever
from src.tools.web_search import tavily_search


# ──────────────────────────────────────────────
# Lazy singletons
# ──────────────────────────────────────────────
_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        settings = get_settings()
        _llm = ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            temperature=0,
        )
    return _llm


# ══════════════════════════════════════════════
# Node: Retrieve
# ══════════════════════════════════════════════

def retrieve(state: GraphState) -> dict:
    """Perform hybrid retrieval (dense + sparse) with RRF fusion."""
    question = state.get("rewritten_question") or state["question"]
    logger.info(f"[retrieve] query: {question[:80]}…")

    dense = DenseRetriever()
    sparse = SparseRetriever()

    dense_results = dense.search(question)
    sparse_results = sparse.search(question)
    fused = reciprocal_rank_fusion([dense_results, sparse_results])

    return {"documents": fused}


# ══════════════════════════════════════════════
# Node: Grade Relevance
# ══════════════════════════════════════════════

GRADER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a relevance grader. Given a user question and a retrieved document, "
        "respond with a JSON object: {{\"score\": <float 0-1>}}. "
        "1.0 means perfectly relevant, 0.0 means completely irrelevant.",
    ),
    (
        "human",
        "Question: {question}\n\nDocument:\n{document}",
    ),
])


def grade_documents(state: GraphState) -> dict:
    """Score each document's relevance to the question."""
    settings = get_settings()
    question = state.get("rewritten_question") or state["question"]
    documents = state["documents"]
    llm = _get_llm()

    scores: list[float] = []
    relevant_docs = []

    for doc in documents:
        chain = GRADER_PROMPT | llm | StrOutputParser()
        result = chain.invoke({"question": question, "document": doc.page_content})
        score = _parse_score(result, default=0.0)

        scores.append(score)
        if score >= settings.relevance_threshold:
            relevant_docs.append(doc)

    # Decide routing
    avg_score = sum(scores) / len(scores) if scores else 0.0
    if avg_score >= settings.relevance_threshold or len(relevant_docs) >= 2:
        decision = "relevant"
    elif avg_score >= settings.relevance_threshold * 0.5 or len(relevant_docs) >= 1:
        decision = "ambiguous"
    else:
        decision = "irrelevant"

    logger.info(f"[grade] avg_score={avg_score:.3f} → {decision} ({len(relevant_docs)}/{len(documents)} pass)")

    return {
        "documents": relevant_docs,  # only relevant docs — no fallback to irrelevant ones
        "relevance_scores": scores,
        "relevance_decision": decision,
    }


# ══════════════════════════════════════════════
# Node: Rewrite Query
# ══════════════════════════════════════════════

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a query rewriter. Given a question that yielded poor retrieval results, "
        "rewrite it to be more specific and search-friendly. "
        "Return ONLY the rewritten question, nothing else.",
    ),
    ("human", "Original question: {question}"),
])


def rewrite_query(state: GraphState) -> dict:
    """Rewrite the query to improve retrieval quality."""
    question = state["question"]
    llm = _get_llm()
    chain = REWRITE_PROMPT | llm | StrOutputParser()
    rewritten = chain.invoke({"question": question})

    logger.info(f"[rewrite] '{question[:50]}…' → '{rewritten[:50]}…'")
    return {
        "rewritten_question": rewritten.strip(),
        "retry_count": state.get("retry_count", 0) + 1,
    }


# ══════════════════════════════════════════════
# Node: Web Search Fallback
# ══════════════════════════════════════════════

def web_search(state: GraphState) -> dict:
    """Fall back to Tavily web search when retrieval fails."""
    question = state.get("rewritten_question") or state["question"]
    logger.info(f"[web_search] query: {question[:80]}…")

    results = tavily_search(question)
    return {
        "web_results": results,
        "documents": results,  # clean web results only — don't carry over irrelevant local docs
    }


# ══════════════════════════════════════════════
# Node: Generate Answer
# ══════════════════════════════════════════════

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert technical assistant. Answer the question directly, accurately, and comprehensively using ONLY the provided context.\n"
        "- Ground every statement strictly in the context.\n"
        "- Directly address the core concepts asked without conversational filler, preamble, or unrelated tangents.\n"
        "- If the context does not contain enough information, state clearly what is supported.",
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion: {question}",
    ),
])


def generate(state: GraphState) -> dict:
    """Generate an answer grounded in the retrieved documents."""
    question = state.get("rewritten_question") or state["question"]
    documents = state["documents"]
    llm = _get_llm()

    context = "\n\n---\n\n".join(doc.page_content for doc in documents)
    chain = GENERATE_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    logger.info(f"[generate] answer length: {len(answer)} chars")
    return {"generation": answer}


# ══════════════════════════════════════════════
# Node: Hallucination Guard
# ══════════════════════════════════════════════

HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a hallucination grader. Given a set of source documents and a generated answer, "
        "determine whether the answer is grounded in the sources. "
        "Respond with a JSON object: {{\"score\": <float 0-1>}}. "
        "1.0 means fully grounded, 0.0 means completely hallucinated.",
    ),
    (
        "human",
        "Sources:\n{sources}\n\nGenerated Answer:\n{answer}",
    ),
])


def hallucination_guard(state: GraphState) -> dict:
    """Check if the generated answer is grounded in source documents."""
    documents = state["documents"]
    answer = state["generation"]
    llm = _get_llm()

    sources = "\n\n---\n\n".join(doc.page_content for doc in documents)
    chain = HALLUCINATION_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"sources": sources, "answer": answer})
    score = _parse_score(result, default=0.0)

    logger.info(f"[hallucination_guard] groundedness score: {score:.3f}")
    return {"hallucination_score": score}
