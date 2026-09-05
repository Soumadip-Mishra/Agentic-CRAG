"""
Text chunking strategies.

Implements Parent-Child chunking and Recursive Character Text Splitting
for optimal retrieval granularity.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

if TYPE_CHECKING:
    from langchain_core.documents import Document


# ──────────────────────────────────────────────
# Recursive Character Chunker
# ──────────────────────────────────────────────

def recursive_chunk(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents using Recursive Character Text Splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Recursive chunking: {len(documents)} docs → {len(chunks)} chunks")
    return chunks


# ──────────────────────────────────────────────
# Parent-Child Chunker
# ──────────────────────────────────────────────

def parent_child_chunk(
    documents: list[Document],
    parent_chunk_size: int = 2000,
    parent_chunk_overlap: int = 200,
    child_chunk_size: int = 500,
    child_chunk_overlap: int = 100,
) -> tuple[list[Document], list[Document]]:
    """
    Create a two-tier chunking hierarchy.

    Returns:
        (parent_chunks, child_chunks) — each child carries a
        ``parent_id`` in its metadata for lookup.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
    )

    parent_chunks = parent_splitter.split_documents(documents)

    # Tag each parent with a unique ID
    for parent in parent_chunks:
        parent.metadata["parent_id"] = str(uuid.uuid4())

    # Split each parent into children
    child_chunks: list[Document] = []
    for parent in parent_chunks:
        children = child_splitter.split_documents([parent])
        for child in children:
            child.metadata["parent_id"] = parent.metadata["parent_id"]
            child.metadata["parent_content"] = parent.page_content
        child_chunks.extend(children)

    logger.info(
        f"Parent-child chunking: {len(documents)} docs → "
        f"{len(parent_chunks)} parents, {len(child_chunks)} children"
    )
    return parent_chunks, child_chunks
