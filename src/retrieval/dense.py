"""
Dense retrieval via Qdrant vector database.

Wraps the Qdrant client to perform cosine-similarity search
against pre-indexed document embeddings.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from loguru import logger
from qdrant_client import QdrantClient

from config.settings import get_settings


class DenseRetriever:
    """Retrieve documents from Qdrant using dense vector search."""

    def __init__(self) -> None:
        settings = get_settings()
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            check_compatibility=False,
        )
        self.collection = settings.qdrant_collection
        self.embedder = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )

    def search(self, query: str, top_k: int = 10) -> list[Document]:
        """Embed the query and search Qdrant for nearest neighbours."""
        query_vector = self.embedder.embed_query(query)

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
        )

        documents = []
        for hit in results.points:
            payload = hit.payload or {}
            content = payload.get("parent_content") or payload.get("content", "")
            metadata = {
                **{k: v for k, v in payload.items() if k not in ("content", "parent_content")},
                "child_content": payload.get("content", ""),
                "score": hit.score,
                "retriever": "dense",
            }
            if "parent_id" in payload and payload["parent_id"]:
                metadata["parent_id"] = payload["parent_id"]

            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

        logger.debug(f"Dense search returned {len(documents)} results for: {query[:60]}…")
        return documents
