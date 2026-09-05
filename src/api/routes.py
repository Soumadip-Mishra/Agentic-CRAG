"""
API route definitions.

Endpoints:
    POST /api/v1/chat    — Query the CRAG agent
    POST /api/v1/ingest  — Trigger document ingestion
    GET  /health         — Health check
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

router = APIRouter()


# ──────────────────────────────────────────────
# Request / Response Schemas
# ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []
    relevance_decision: str = ""
    hallucination_score: float = 0.0
    retries: int = 0


class IngestRequest(BaseModel):
    data_dir: str = Field(default="data/raw", description="Path to documents directory")


class IngestResponse(BaseModel):
    status: str
    documents_processed: int = 0


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Query the CRAG agent and receive a self-corrected answer."""
    try:
        from src.agent.workflow import app as crag_app

        initial_state = {
            "question": request.query,
            "rewritten_question": "",
            "documents": [],
            "web_results": [],
            "relevance_scores": [],
            "relevance_decision": "",
            "generation": "",
            "hallucination_score": 0.0,
            "retry_count": 0,
            "route": "",
        }

        result = crag_app.invoke(initial_state)

        sources = list({
            doc.metadata.get("source", "unknown")
            for doc in result.get("documents", [])
        })

        return ChatResponse(
            answer=result.get("generation", ""),
            sources=sources,
            relevance_decision=result.get("relevance_decision", ""),
            hallucination_score=result.get("hallucination_score", 0.0),
            retries=result.get("retry_count", 0),
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """Trigger the document ingestion pipeline."""
    try:
        from src.ingestion.pipeline import run_pipeline

        data_path = Path(request.data_dir)
        if not data_path.exists():
            raise HTTPException(status_code=404, detail=f"Directory not found: {request.data_dir}")

        processed = run_pipeline(data_dir=data_path)
        return IngestResponse(status="success", documents_processed=processed)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingest endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "agentic-crag"}
