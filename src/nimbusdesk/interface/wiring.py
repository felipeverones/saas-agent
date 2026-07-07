"""The shared composition root — where every port meets its adapter.

Both entry points (the dev CLI in agents/__main__.py and the FastAPI service)
need the same wiring: settings -> providers (traced + usage-tracked) ->
retrieval stack -> memory -> compiled graph with a SQLite checkpointer.
Defining it once here means the API and the CLI can never drift apart, and
tests can build the same runtime with fakes swapped in at any seam.
"""

import logging
import sqlite3
from dataclasses import dataclass

from qdrant_client import QdrantClient

from nimbusdesk.infrastructure.anthropic_llm import AnthropicProvider
from nimbusdesk.infrastructure.embeddings import FastEmbedEmbedder, FastEmbedSparseEmbedder
from nimbusdesk.infrastructure.reranker import FastEmbedReranker
from nimbusdesk.infrastructure.settings import Settings
from nimbusdesk.infrastructure.vector_store import QdrantVectorIndex
from nimbusdesk.llm.tracking import UsageTracker
from nimbusdesk.memory.episodic import EpisodicMemoryStore
from nimbusdesk.memory.profile_store import SqliteProfileStore
from nimbusdesk.memory.service import MemoryService
from nimbusdesk.memory.writer import MemoryWriter
from nimbusdesk.observability.llm import TracingLLM
from nimbusdesk.rag.retrieval import Retriever

logger = logging.getLogger(__name__)


@dataclass
class AppRuntime:
    """Everything a serving process needs, built once at startup."""

    graph: object  # CompiledStateGraph
    fast: UsageTracker
    strong: UsageTracker
    settings: Settings


def build_llms(settings: Settings) -> tuple[UsageTracker, UsageTracker]:
    api_key = (
        settings.anthropic_api_key.get_secret_value()
        if settings.anthropic_api_key
        else None
    )
    # Decorator stack on one port: provider -> tracing spans -> token
    # accounting. Each layer is oblivious to the others (phase 2's promise).
    fast = UsageTracker(
        TracingLLM(AnthropicProvider(api_key=api_key, model=settings.nimbus_model_fast))
    )
    strong = UsageTracker(
        TracingLLM(AnthropicProvider(api_key=api_key, model=settings.nimbus_model_strong))
    )
    return fast, strong


def build_retrieval(settings: Settings) -> tuple[Retriever, FastEmbedReranker]:
    retriever = Retriever(
        FastEmbedEmbedder(settings.embedding_model_name, settings.embedding_dimension),
        FastEmbedSparseEmbedder(settings.sparse_model_name),
        QdrantVectorIndex(
            client=QdrantClient(url=settings.qdrant_url),
            collection=settings.qdrant_collection,
            dimension=settings.embedding_dimension,
        ),
    )
    return retriever, FastEmbedReranker(settings.reranker_model_name)


def build_memory(settings: Settings, fast_llm) -> MemoryService:
    profiles = SqliteProfileStore(
        sqlite3.connect(settings.memory_db_path, check_same_thread=False)
    )
    episodes = EpisodicMemoryStore(
        client=QdrantClient(url=settings.qdrant_url),
        embedder=FastEmbedEmbedder(
            settings.embedding_model_name, settings.embedding_dimension
        ),
        collection=settings.memory_collection,
    )
    return MemoryService(profiles, episodes, MemoryWriter(fast_llm, profiles, episodes))


def ensure_knowledge_base(settings: Settings) -> None:
    """First-boot convenience: if the KB collection doesn't exist yet, ingest
    the seed corpus so `docker compose up` works with zero manual steps."""
    client = QdrantClient(url=settings.qdrant_url)
    if client.collection_exists(settings.qdrant_collection):
        return
    from pathlib import Path

    from nimbusdesk.rag.ingestion import IngestionPipeline

    logger.info("knowledge base empty — ingesting data/seed")
    dense = FastEmbedEmbedder(settings.embedding_model_name, settings.embedding_dimension)
    sparse = FastEmbedSparseEmbedder(settings.sparse_model_name)
    index = QdrantVectorIndex(client, settings.qdrant_collection, settings.embedding_dimension)
    report = IngestionPipeline(dense, sparse, index).run(Path("data/seed"))
    logger.info("ingested %d documents as %d chunks", report.documents, report.chunks)


def maybe_enable_tracing(settings: Settings) -> None:
    if settings.tracing_enabled:
        from nimbusdesk.observability.tracing import setup_tracing

        setup_tracing(settings.otel_exporter_otlp_endpoint)
        logger.info("tracing to %s", settings.otel_exporter_otlp_endpoint)


def build_runtime(settings: Settings, account_tools=None) -> AppRuntime:
    from langgraph.checkpoint.sqlite import SqliteSaver

    from nimbusdesk.agents.graph import build_support_graph

    fast, strong = build_llms(settings)
    retriever, reranker = build_retrieval(settings)
    # check_same_thread=False: LangGraph may touch the connection from worker
    # threads; SQLite forbids cross-thread use by default.
    checkpointer = SqliteSaver(
        sqlite3.connect(settings.checkpoint_db_path, check_same_thread=False)
    )
    graph = build_support_graph(
        fast,
        strong,
        retriever,
        reranker,
        checkpointer,
        account_tools=account_tools,
        memory=build_memory(settings, fast),
    )
    return AppRuntime(graph=graph, fast=fast, strong=strong, settings=settings)
