"""Application settings — all environment config, validated at startup.

WHY pydantic-settings: configuration errors should kill the process at boot
with a readable message, not surface as a cryptic failure mid-conversation.
Every field maps to an env var of the same name in UPPERCASE (loaded from the
environment or a local `.env` file — see `.env.example` for documentation).
"""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Optional here because phases 1-2 run without an LLM. The Anthropic
    # adapter (phase 3) raises at construction time if it's missing — the
    # "fail at startup" rule enforced at the point where the key is needed.
    anthropic_api_key: SecretStr | None = None
    nimbus_model_strong: str = "claude-sonnet-5"
    nimbus_model_fast: str = "claude-haiku-4-5-20251001"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "nimbus_kb"
    # Long-term memory stores (phase 6)
    memory_collection: str = "nimbus_memories"
    memory_db_path: str = "data/memory.sqlite"

    # bge-small-en-v1.5: 384-dim English model, ~65 MB, runs on CPU via ONNX.
    # Small on purpose — free ingestion and fast tests; swapping for a larger
    # model is a settings change, not a code change (see ADR-04).
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    # BM25 term weights for the sparse channel of hybrid search (ADR-03).
    sparse_model_name: str = "Qdrant/bm25"
    # Small cross-encoder (~23 MB) for reranking the candidate funnel (ADR-05).
    reranker_model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Retrieval funnel shape: fetch generously (recall), rerank down (precision).
    retrieval_candidates: int = 20
    retrieval_top_k: int = 5

    otel_exporter_otlp_endpoint: str = "http://localhost:4317"

    # MCP servers (phase 5) — each runs as its own process (`make mcp-crm`,
    # `make mcp-ticketing`); agents reach them over Streamable HTTP.
    mcp_crm_url: str = "http://localhost:8101/mcp"
    mcp_ticketing_url: str = "http://localhost:8102/mcp"


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so every module shares one validated instance.

    Import-time singletons (`settings = Settings()` at module level) make
    testing painful — a function behind lru_cache is trivially overridable.
    """
    return Settings()
