"""
Sparse retrieval using BM25 (Okapi BM25).

Loads the pre-built BM25 index and corpus from disk to
perform keyword-based document retrieval.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from loguru import logger
from rank_bm25 import BM25Okapi

BM25_INDEX_PATH = Path("data/bm25_index.pkl")
BM25_CORPUS_PATH = Path("data/bm25_corpus.json")


class SparseRetriever:
    """Retrieve documents using BM25 keyword matching."""

    def __init__(self) -> None:
        if not BM25_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {BM25_INDEX_PATH}. "
                "Run the ingestion pipeline first."
            )

        with open(BM25_INDEX_PATH, "rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)

        with open(BM25_CORPUS_PATH) as f:
            self.corpus: list[dict] = json.load(f)

        logger.info(f"BM25 index loaded with {len(self.corpus)} documents")

    def search(self, query: str, top_k: int = 10) -> list[Document]:
        """Score all documents against the query and return top-k."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        documents = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            entry = self.corpus[idx]
            content = entry.get("parent_content") or entry.get("content", "")
            metadata = {
                **entry.get("metadata", {}),
                "child_content": entry.get("content", ""),
                "score": float(scores[idx]),
                "retriever": "sparse",
            }
            if "parent_id" in entry and entry["parent_id"]:
                metadata["parent_id"] = entry["parent_id"]

            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

        logger.debug(f"BM25 search returned {len(documents)} results for: {query[:60]}…")
        return documents
