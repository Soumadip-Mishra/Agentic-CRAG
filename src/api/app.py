"""
FastAPI application initialization & middleware.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config.settings import get_settings
from src.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup & shutdown hooks."""
    settings = get_settings()
    logger.info(f"🚀 Agentic CRAG API starting on {settings.api_host}:{settings.api_port}")
    yield
    logger.info("👋 Agentic CRAG API shutting down")


app = FastAPI(
    title="Agentic CRAG API",
    description="Corrective Retrieval-Augmented Generation with agentic self-correction.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS Middleware ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routes ──
app.include_router(router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
async def root_health():
    """Root health check for Docker, load balancers, and monitoring."""
    return {"status": "healthy", "service": "agentic-crag"}

