"""
Tavily Search API fallback wrapper.

Used when local retrieval fails to find relevant documents,
providing a web-search safety net.
"""

from __future__ import annotations

from langchain_core.documents import Document
from loguru import logger

from config.settings import get_settings


def tavily_search(query: str, max_results: int = 5) -> list[Document]:
    """
    Search the web via Tavily and return results as LangChain Documents.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of Document objects with web content and source URLs.
    """
    settings = get_settings()

    if not settings.tavily_api_key:
        logger.warning("Tavily API key not set — returning empty results")
        return []

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(query=query, max_results=max_results)

        documents = []
        for result in response.get("results", []):
            documents.append(
                Document(
                    page_content=result.get("content", ""),
                    metadata={
                        "source": result.get("url", ""),
                        "title": result.get("title", ""),
                        "retriever": "web_search",
                    },
                )
            )

        logger.info(f"Tavily returned {len(documents)} results for: {query[:60]}…")
        return documents

    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return []
