"""
Multi-format document parsers.

Supports PDF, DOCX, Markdown, and plain-text files via
PyMuPDF, python-docx, and the Unstructured library.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from langchain_core.documents import Document


# ──────────────────────────────────────────────
# Individual Parsers
# ──────────────────────────────────────────────

def parse_pdf(file_path: str | Path) -> list[Document]:
    """Extract text from a PDF using PyMuPDF."""
    from langchain_core.documents import Document

    import pymupdf

    docs: list[Document] = []
    file_path = Path(file_path)
    logger.info(f"Parsing PDF: {file_path.name}")

    with pymupdf.open(str(file_path)) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text()
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path),
                            "page": page_num,
                            "format": "pdf",
                        },
                    )
                )
    logger.info(f"  → extracted {len(docs)} pages")
    return docs


def parse_docx(file_path: str | Path) -> list[Document]:
    """Extract text from a DOCX file."""
    from langchain_core.documents import Document
    from docx import Document as DocxDocument

    file_path = Path(file_path)
    logger.info(f"Parsing DOCX: {file_path.name}")

    doc = DocxDocument(str(file_path))
    full_text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())

    return [
        Document(
            page_content=full_text,
            metadata={"source": str(file_path), "format": "docx"},
        )
    ]


def parse_markdown(file_path: str | Path) -> list[Document]:
    """Read a Markdown file as a single document."""
    from langchain_core.documents import Document

    file_path = Path(file_path)
    logger.info(f"Parsing Markdown: {file_path.name}")

    text = file_path.read_text(encoding="utf-8")
    return [
        Document(
            page_content=text,
            metadata={"source": str(file_path), "format": "markdown"},
        )
    ]


def parse_text(file_path: str | Path) -> list[Document]:
    """Read a plain-text file as a single document."""
    from langchain_core.documents import Document

    file_path = Path(file_path)
    logger.info(f"Parsing TXT: {file_path.name}")

    text = file_path.read_text(encoding="utf-8")
    return [
        Document(
            page_content=text,
            metadata={"source": str(file_path), "format": "txt"},
        )
    ]


# ──────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────

PARSER_REGISTRY: dict[str, callable] = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".md": parse_markdown,
    ".txt": parse_text,
}


def parse_file(file_path: str | Path) -> list[Document]:
    """Route a file to the appropriate parser based on extension."""
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    parser = PARSER_REGISTRY.get(ext)
    if parser is None:
        logger.warning(f"No parser registered for extension '{ext}', skipping {file_path.name}")
        return []

    return parser(file_path)


def parse_directory(directory: str | Path) -> list[Document]:
    """Parse all supported files in a directory (non-recursive)."""
    directory = Path(directory)
    all_docs: list[Document] = []

    for file_path in sorted(directory.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in PARSER_REGISTRY:
            all_docs.extend(parse_file(file_path))

    logger.info(f"Total documents parsed from {directory}: {len(all_docs)}")
    return all_docs
