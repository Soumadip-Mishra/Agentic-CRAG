"""
Unit & integration tests for the retrieval subsystem.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.retrieval.fusion import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    """Tests for the RRF fusion logic."""

    def test_single_list_preserves_order(self):
        """A single result list should maintain its original ranking."""
        docs = [
            Document(page_content="doc A", metadata={}),
            Document(page_content="doc B", metadata={}),
            Document(page_content="doc C", metadata={}),
        ]
        fused = reciprocal_rank_fusion([docs])
        assert len(fused) == 3
        assert fused[0].page_content == "doc A"

    def test_two_lists_merge_correctly(self):
        """Documents appearing in both lists should rank higher."""
        list1 = [
            Document(page_content="shared doc", metadata={}),
            Document(page_content="only in list1", metadata={}),
        ]
        list2 = [
            Document(page_content="only in list2", metadata={}),
            Document(page_content="shared doc", metadata={}),
        ]
        fused = reciprocal_rank_fusion([list1, list2])
        # Shared doc should be first (appears in both lists)
        assert fused[0].page_content == "shared doc"

    def test_empty_lists(self):
        """Empty input should return empty output."""
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[]]) == []

    def test_rrf_scores_are_positive(self):
        """All RRF scores should be positive floats."""
        docs = [Document(page_content=f"doc {i}", metadata={}) for i in range(5)]
        fused = reciprocal_rank_fusion([docs])
        for doc in fused:
            assert doc.metadata["rrf_score"] > 0


class TestSparseRetriever:
    """Smoke tests for BM25 sparse retriever (requires ingested data)."""

    @pytest.mark.skipif(
        not __import__("pathlib").Path("data/bm25_index.pkl").exists(),
        reason="BM25 index not built — run ingestion pipeline first",
    )
    def test_search_returns_documents(self):
        """BM25 search should return non-empty results for a known query."""
        from src.retrieval.sparse import SparseRetriever

        retriever = SparseRetriever()
        results = retriever.search("test query", top_k=3)
        assert isinstance(results, list)

    def test_sparse_resolves_parent_content(self, monkeypatch, tmp_path):
        """SparseRetriever should unpack parent_content as page_content when available."""
        import json
        import pickle
        from rank_bm25 import BM25Okapi
        from src.retrieval import sparse

        # Setup mock index and corpus
        test_index_path = tmp_path / "bm25_index.pkl"
        test_corpus_path = tmp_path / "bm25_corpus.json"

        bm25 = BM25Okapi([["child", "chunk", "keywords"], ["other", "unrelated", "doc"], ["third", "sample", "document"]])
        with open(test_index_path, "wb") as f:
            pickle.dump(bm25, f)

        corpus = [
            {
                "content": "child chunk keywords",
                "parent_content": "Full parent text with detailed information.",
                "parent_id": "parent-123",
                "metadata": {"source": "doc.pdf"},
            },
            {
                "content": "other unrelated doc",
                "parent_content": "Other parent text.",
                "parent_id": "parent-456",
                "metadata": {"source": "doc2.pdf"},
            },
            {
                "content": "third sample document",
                "parent_content": "Third parent text.",
                "parent_id": "parent-789",
                "metadata": {"source": "doc3.pdf"},
            },
        ]
        with open(test_corpus_path, "w", encoding="utf-8") as f:
            json.dump(corpus, f)

        monkeypatch.setattr(sparse, "BM25_INDEX_PATH", test_index_path)
        monkeypatch.setattr(sparse, "BM25_CORPUS_PATH", test_corpus_path)

        retriever = sparse.SparseRetriever()
        docs = retriever.search("child", top_k=1)

        assert len(docs) == 1
        assert docs[0].page_content == "Full parent text with detailed information."
        assert docs[0].metadata["child_content"] == "child chunk keywords"
        assert docs[0].metadata["parent_id"] == "parent-123"
