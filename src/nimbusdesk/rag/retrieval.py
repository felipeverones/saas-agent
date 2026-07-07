"""Retrieval — answering "which chunks are relevant to this question?".

Phase 2 upgrade: HYBRID search. The query is embedded through two independent
channels — dense (semantic meaning) and sparse (exact BM25 term weights) — and
the index fuses both rankings server-side (RRF, see vector_store.py).

WHY BOTH, IN ONE EXAMPLE EACH
- "customer wants money back"  -> dense wins: no word overlap with "refund
  policy", but the meanings are neighbors in embedding space.
- "error ND-WH-TLS"            -> sparse wins: rare identifiers are exactly
  what embedding models blur, and exactly what BM25 nails.
Real support traffic is a mix of both phrasings; shipping only one channel
means silently failing half your users.
"""

from nimbusdesk.domain.knowledge import RetrievedChunk
from nimbusdesk.observability.tracing import span
from nimbusdesk.rag.ports import Embedder, SparseEmbedder, VectorIndex

DEFAULT_TOP_K = 5


class Retriever:
    def __init__(
        self, embedder: Embedder, sparse_embedder: SparseEmbedder, index: VectorIndex
    ) -> None:
        self._embedder = embedder
        self._sparse_embedder = sparse_embedder
        self._index = index

    def search(self, query: str, k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        with span("rag.retrieve", query=query[:200], k=k) as current:
            # embed_query, not embed_passages: retrieval models are asymmetric
            # (see ports.py) — mixing the two silently degrades quality.
            dense = self._embedder.embed_query(query)
            sparse = self._sparse_embedder.embed_query(query)
            results = self._index.search(dense, sparse, k)
            current.set_attribute("rag.results", len(results))
            current.set_attribute(
                "rag.result_docs", ",".join(r.chunk.doc_id for r in results[:10])
            )
            return results
