"""
Unit tests for parent-child chunking and retrieval resolution.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.ingestion.chunker import parent_child_chunk
from src.retrieval.fusion import reciprocal_rank_fusion


class TestParentChildChunker:
    """Tests for parent-child chunking hierarchy and metadata tagging."""

    def test_parent_child_creates_hierarchy(self):
        long_text = "Paragraph one content. " * 50 + "\n\n" + "Paragraph two content. " * 50
        doc = Document(page_content=long_text, metadata={"source": "test.txt", "author": "Alice"})

        parents, children = parent_child_chunk(
            [doc],
            parent_chunk_size=500,
            parent_chunk_overlap=50,
            child_chunk_size=150,
            child_chunk_overlap=30,
        )

        assert len(parents) > 1
        assert len(children) > len(parents)

        # Verify parent IDs and parent content attachment
        parent_ids = {p.metadata["parent_id"] for p in parents}
        assert len(parent_ids) == len(parents)

        for child in children:
            assert "parent_id" in child.metadata
            assert child.metadata["parent_id"] in parent_ids
            assert "parent_content" in child.metadata
            assert child.metadata["source"] == "test.txt"
            assert child.metadata["author"] == "Alice"
            # Child content must be a substring of parent content
            assert child.page_content in child.metadata["parent_content"]


class TestParentChildRRF:
    """Tests for RRF fusion with parent_id deduplication."""

    def test_deduplicates_by_parent_id(self):
        # Two children from the same parent retrieved by different retrievers
        parent_text = "Full parent text containing comprehensive explanation."
        doc_dense = Document(
            page_content=parent_text,
            metadata={"parent_id": "parent-uuid-1", "retriever": "dense"},
        )
        doc_sparse = Document(
            page_content=parent_text,
            metadata={"parent_id": "parent-uuid-1", "retriever": "sparse"},
        )
        doc_other = Document(
            page_content="Another unrelated document",
            metadata={"parent_id": "parent-uuid-2", "retriever": "dense"},
        )

        fused = reciprocal_rank_fusion([[doc_dense, doc_other], [doc_sparse]])

        # parent-uuid-1 should be deduplicated to a single result and ranked #1
        assert len(fused) == 2
        assert fused[0].metadata["parent_id"] == "parent-uuid-1"
        assert fused[1].metadata["parent_id"] == "parent-uuid-2"
        # The deduplicated parent accumulated scores from both lists
        assert fused[0].metadata["rrf_score"] > fused[1].metadata["rrf_score"]

