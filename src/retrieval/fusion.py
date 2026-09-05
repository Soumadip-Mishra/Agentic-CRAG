"""
Reciprocal Rank Fusion (RRF).

Merges ranked result lists from multiple retrievers into a
single, unified ranking using the RRF formula, then returns the
top-N highest-scoring documents.
"""

from __future__ import annotations

from langchain_core.documents import Document
from loguru import logger

from config.settings import get_settings


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    k: int = 60,
    top_n: int | None = None,
) -> list[Document]:
    """
    Fuse multiple ranked document lists using RRF.

    RRF score for a document d:
        RRF(d) = Σ  1 / (k + rank_i(d))

    Args:
        result_lists: List of ranked document lists from different retrievers.
        k: Constant to prevent high-ranked documents from dominating (default 60).
        top_n: Number of top-ranked documents to return. Defaults to
               ``settings.retrieval_top_n`` (5).

    Returns:
        Top-N documents sorted by descending RRF score.
    """
    if top_n is None:
        top_n = get_settings().retrieval_top_n

    fused_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            # Use parent_id if available, otherwise first 200 chars as dedup key
            doc_key = doc.metadata.get("parent_id") or doc.page_content[:200]
            if doc_key not in doc_map:
                doc_map[doc_key] = doc
                fused_scores[doc_key] = 0.0
            fused_scores[doc_key] += 1.0 / (k + rank)

    # Sort by fused score descending
    sorted_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)
    fused_docs = []
    for key in sorted_keys:
        doc = doc_map[key]
        doc.metadata["rrf_score"] = fused_scores[key]
        fused_docs.append(doc)

    top_docs = fused_docs[:top_n]
    logger.debug(
        f"RRF fused {sum(len(r) for r in result_lists)} results "
        f"→ {len(fused_docs)} unique → top {len(top_docs)}"
    )
    return top_docs
