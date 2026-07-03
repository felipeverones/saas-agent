"""Qdrant adapter — the concrete VectorIndex behind the port in rag/ports.py.

DESIGN NOTES
- The QdrantClient is INJECTED, not created here: production wires
  `QdrantClient(url=...)`, tests wire `QdrantClient(":memory:")` (a full
  in-process Qdrant) — same adapter code exercised in both.
- HYBRID SEARCH lives entirely in this adapter: each chunk is stored under a
  named dense vector ("dense", semantic) AND a named sparse vector ("sparse",
  BM25 term weights). At query time we prefetch candidates from both channels
  and let Qdrant fuse them server-side with Reciprocal Rank Fusion (RRF).
- WHY RRF: dense scores (cosine ~0..1) and BM25 scores (unbounded) live on
  incomparable scales — averaging them is meaningless. RRF ignores scores and
  combines RANKS (1/(rank+const) from each list), which is scale-free, robust,
  and the industry-default fusion for exactly this reason.
- Cosine distance because our embedding model produces normalized vectors
  trained for cosine similarity — distance metric must match the model.
"""

from typing import Sequence

from qdrant_client import QdrantClient, models

from nimbusdesk.domain.knowledge import DocumentChunk, RetrievedChunk
from nimbusdesk.rag.ports import SparseVector

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# How many candidates each channel contributes before fusion. Generous on
# purpose: recall lost here is unrecoverable downstream (the reranker can only
# promote what retrieval surfaced).
PREFETCH_PER_CHANNEL = 20


class QdrantVectorIndex:
    def __init__(self, client: QdrantClient, collection: str, dimension: int) -> None:
        self._client = client
        self._collection = collection
        self._dimension = dimension

    def ensure_ready(self) -> None:
        if self._client.collection_exists(self._collection):
            if self._has_sparse_schema():
                return
            # Schema migration: the phase-1 collection had no sparse vectors.
            # Drop-and-recreate is fine HERE because the index is derived data,
            # fully rebuildable from data/seed. In production you'd build a new
            # collection alongside and flip a collection ALIAS atomically
            # (blue/green), so search never sees an empty index.
            self._client.delete_collection(self._collection)

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=self._dimension, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                # IDF modifier: Qdrant scales matches by term rarity server-side
                # (the "IDF" in BM25). Without it, "the" and "ND-WH-TLS" would
                # count the same and lexical search would be noise.
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
        )

    def _has_sparse_schema(self) -> bool:
        info = self._client.get_collection(self._collection)
        return SPARSE_VECTOR_NAME in (info.config.params.sparse_vectors or {})

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        dense: Sequence[list[float]],
        sparse: Sequence[SparseVector],
    ) -> None:
        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector={
                    DENSE_VECTOR_NAME: dense_vec,
                    SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=sparse_vec.indices, values=sparse_vec.values
                    ),
                },
                # The full chunk goes into the payload: search results must be
                # reconstructable into domain objects without a second lookup.
                payload=chunk.model_dump(),
            )
            for chunk, dense_vec, sparse_vec in zip(chunks, dense, sparse, strict=True)
        ]
        # wait=True: don't report success until points are durably applied.
        self._client.upsert(collection_name=self._collection, points=points, wait=True)

    def search(self, dense: list[float], sparse: SparseVector, k: int) -> list[RetrievedChunk]:
        result = self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                models.Prefetch(
                    query=dense, using=DENSE_VECTOR_NAME, limit=PREFETCH_PER_CHANNEL
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                    using=SPARSE_VECTOR_NAME,
                    limit=PREFETCH_PER_CHANNEL,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
            with_payload=True,
        )
        return [
            RetrievedChunk(chunk=DocumentChunk.model_validate(p.payload), score=p.score)
            for p in result.points
        ]
