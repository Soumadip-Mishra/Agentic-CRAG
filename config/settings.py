"""
Centralized application settings.

All configuration is loaded from environment variables (via .env)
and validated through Pydantic BaseSettings.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM Provider ──
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── Vector Database (Qdrant) ──
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "agentic_crag"
    qdrant_api_key: str = ""

    # ── Redis Cache ──
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""

    # ── Web Search Fallback (Tavily) ──
    tavily_api_key: str = ""

    # ── Agent Thresholds ──
    relevance_threshold: float = 0.7
    hallucination_threshold: float = 0.8
    max_retries: int = 3
    retrieval_top_n: int = 5

    # ── Chunking Strategy ──
    chunking_strategy: str = "parent_child"  # "parent_child" | "recursive"
    parent_chunk_size: int = 2000
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 500
    child_chunk_overlap: int = 100

    # ── FastAPI Server ──
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "info"

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/0"
        return f"redis://{self.redis_host}:{self.redis_port}/0"


@lru_cache()
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    return Settings()
