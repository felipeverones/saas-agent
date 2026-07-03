"""Ports (interfaces) the RAG pipeline depends on.

CONCEPT: DEPENDENCY INVERSION IN PRACTICE
This module is the "wall socket": the RAG code (application layer) declares
WHAT it needs — something that turns text into vectors, something that stores
and searches them — without naming any vendor. The concrete "plugs"
(FastEmbed, Qdrant) live in `infrastructure/` and implement these Protocols.

WHY `typing.Protocol` INSTEAD OF ABC BASE CLASSES
Protocols are structural: any class with matching methods satisfies the port,
no inheritance required. That keeps infrastructure adapters AND test fakes
completely decoupled from this module — they don't even import it.
"""

from typing import Protocol, Sequence

from pydantic import BaseModel

from nimbusdesk.domain.knowledge import DocumentChunk, RetrievedChunk


class SparseVector(BaseModel):
    """A mostly-zeros vector stored as (index, value) pairs.

    One dimension per vocabulary term; only terms present in the text get an
    entry. This is the data shape of lexical (BM25-style) search — exact-token
    signal, complementary to dense semantic vectors.
    """

    indices: list[int]
    values: list[float]


class Embedder(Protocol):
    """Turns text into fixed-size vectors.

    Passages and queries are embedded through DIFFERENT methods because modern
    retrieval models are asymmetric: they are trained to place a short question
    and the long passage that answers it close together, and some (like our
    bge model) expect a special prefix on queries. Collapsing both into one
    `embed()` is a classic silent-quality-loss bug.
    """

    @property
    def dimension(self) -> int: ...

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SparseEmbedder(Protocol):
    """Turns text into sparse (lexical/BM25) vectors — the exact-token channel."""

    def embed_passages(self, texts: Sequence[str]) -> list[SparseVector]: ...

    def embed_query(self, text: str) -> SparseVector: ...


class VectorIndex(Protocol):
    """Stores chunks under BOTH representations and runs hybrid search.

    Hybrid = dense (meaning) + sparse (exact tokens), fused into one ranking.
    The fusion strategy (RRF) is the adapter's concern — the pipeline only
    knows it hands over both query vectors and gets one ranked list back.
    """

    def ensure_ready(self) -> None:
        """Create (or migrate) the underlying collection/schema."""
        ...

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        dense: Sequence[list[float]],
        sparse: Sequence[SparseVector],
    ) -> None:
        """Insert-or-update. Chunks carry deterministic ids, so re-ingesting the
        same documents overwrites instead of duplicating (idempotency)."""
        ...

    def search(
        self, dense: list[float], sparse: SparseVector, k: int
    ) -> list[RetrievedChunk]: ...


class Reranker(Protocol):
    """Re-scores a candidate set with a more precise (and more expensive) model.

    Sits after retrieval in the funnel: retrieval optimizes RECALL over the
    whole corpus, the reranker optimizes PRECISION over ~20 candidates.
    """

    def rerank(
        self, query: str, results: Sequence[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]: ...
