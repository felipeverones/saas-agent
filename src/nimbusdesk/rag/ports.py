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

from nimbusdesk.domain.knowledge import DocumentChunk, RetrievedChunk


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


class VectorIndex(Protocol):
    """Stores chunk vectors and finds the nearest ones to a query vector."""

    def ensure_ready(self) -> None:
        """Create the underlying collection/schema if it doesn't exist yet."""
        ...

    def upsert(self, chunks: Sequence[DocumentChunk], vectors: Sequence[list[float]]) -> None:
        """Insert-or-update. Chunks carry deterministic ids, so re-ingesting the
        same documents overwrites instead of duplicating (idempotency)."""
        ...

    def search(self, vector: list[float], k: int) -> list[RetrievedChunk]: ...
