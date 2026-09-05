"""
Full ingestion pipeline.

Orchestrates: parse → chunk → embed → store in Qdrant + build BM25 index.
Can be invoked as ``python -m src.ingestion.pipeline``.
"""

from __future__ import annotations

import json
import pickle
import uuid
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from config.settings import get_settings
from src.ingestion.chunker import parent_child_chunk, recursive_chunk
from src.ingestion.parsers import parse_directory

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
DATA_DIR = Path("data/raw")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")
BM25_CORPUS_PATH = Path("data/bm25_corpus.json")


def _build_bm25_index(chunks: list) -> BM25Okapi:
    """Build and persist a BM25 index from chunk texts."""
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenized = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized)

    # Persist the index & corpus for retrieval
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)

    corpus = [
        {
            "content": doc.page_content,
            "parent_content": doc.metadata.get("parent_content", doc.page_content),
            "parent_id": doc.metadata.get("parent_id", ""),
            "metadata": doc.metadata,
        }
        for doc in chunks
    ]
    with open(BM25_CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, default=str)

    logger.info(f"BM25 index saved → {BM25_INDEX_PATH}")
    return bm25


def _store_in_qdrant(chunks: list, embeddings: list[list[float]]) -> None:
    """Upsert embeddings into Qdrant."""
    settings = get_settings()
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        check_compatibility=False,
    )

    # Create collection if missing
    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in collections:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=len(embeddings[0]),
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created Qdrant collection: {settings.qdrant_collection}")

    # Build points with UUIDs and parent metadata
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={
                "content": chunk.page_content,
                "parent_content": chunk.metadata.get("parent_content", chunk.page_content),
                "parent_id": chunk.metadata.get("parent_id", ""),
                **chunk.metadata,
            },
        )
        for chunk, emb in zip(chunks, embeddings)
    ]

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=points[i : i + batch_size],
        )

    logger.info(f"Upserted {len(points)} vectors into Qdrant")


def run_pipeline(data_dir: Path | None = None) -> int:
    """Execute the full ingestion pipeline. Returns count of processed chunks."""
    settings = get_settings()
    data_dir = data_dir or DATA_DIR

    logger.info("═══ Ingestion Pipeline Started ═══")

    # 1. Parse
    documents = parse_directory(data_dir)
    if not documents:
        logger.warning("No documents found. Exiting pipeline.")
        return 0

    # 2. Chunk
    if settings.chunking_strategy == "parent_child":
        logger.info("Using Parent-Child chunking strategy...")
        parent_chunks, child_chunks = parent_child_chunk(
            documents,
            parent_chunk_size=settings.parent_chunk_size,
            parent_chunk_overlap=settings.parent_chunk_overlap,
            child_chunk_size=settings.child_chunk_size,
            child_chunk_overlap=settings.child_chunk_overlap,
        )
        chunks = child_chunks
    else:
        logger.info("Using Recursive chunking strategy...")
        chunks = recursive_chunk(documents)

    # 3. Embed
    logger.info("Generating embeddings …")
    embedder = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key,
    )
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedder.embed_documents(texts)

    # 4. Store in Qdrant
    _store_in_qdrant(chunks, embeddings)

    # 5. Build BM25 index
    _build_bm25_index(chunks)

    logger.info(f"═══ Ingestion Pipeline Complete ({len(chunks)} chunks) ═══")
    return len(chunks)


# Allow running as: python -m src.ingestion.pipeline
if __name__ == "__main__":
    run_pipeline()
