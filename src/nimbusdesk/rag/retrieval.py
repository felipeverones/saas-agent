"""Retrieval — answering "which chunks are relevant to this question?".

Phase 1 keeps this deliberately minimal: embed the query, nearest-neighbor
search, return typed results. It exists as a class (not a bare function)
because phase 2 grows it into the agentic version — query rewriting, hybrid
dense+sparse search, reranking, self-check — without changing its callers.
"""

from nimbusdesk.domain.knowledge import RetrievedChunk
from nimbusdesk.rag.ports import Embedder, VectorIndex

DEFAULT_TOP_K = 5


class Retriever:
    def __init__(self, embedder: Embedder, index: VectorIndex) -> None:
        self._embedder = embedder
        self._index = index

    def search(self, query: str, k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        # embed_query, not embed_passages: retrieval models are asymmetric
        # (see ports.py) — mixing the two silently degrades quality.
        query_vector = self._embedder.embed_query(query)
        return self._index.search(query_vector, k)
