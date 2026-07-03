"""Qdrant adapter — the concrete VectorIndex behind the port in rag/ports.py.

DESIGN NOTES
- The QdrantClient is INJECTED, not created here: production wires
  `QdrantClient(url=...)`, tests wire `QdrantClient(":memory:")` (a full
  in-process Qdrant) — same adapter code exercised in both.
- The collection uses a NAMED vector ("dense") instead of the anonymous
  default. Forward compatibility on purpose: phase 2 adds a "sparse" vector
  for hybrid search to the same collection with zero migration.
- Cosine distance because our embedding model produces normalized vectors
  trained for cosine similarity — distance metric must match the model.
"""

from typing import Sequence

from qdrant_client import QdrantClient, models

from nimbusdesk.domain.knowledge import DocumentChunk, RetrievedChunk

DENSE_VECTOR_NAME = "dense"


class QdrantVectorIndex:
    def __init__(self, client: QdrantClient, collection: str, dimension: int) -> None:
        self._client = client
        self._collection = collection
        self._dimension = dimension

    def ensure_ready(self) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=self._dimension, distance=models.Distance.COSINE
                )
            },
        )

    def upsert(self, chunks: Sequence[DocumentChunk], vectors: Sequence[list[float]]) -> None:
        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector={DENSE_VECTOR_NAME: vector},
                # The full chunk goes into the payload: search results must be
                # reconstructable into domain objects without a second lookup.
                payload=chunk.model_dump(),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        # wait=True: don't report success until points are durably applied —
        # an ingestion CLI that lies about completion is worse than a slow one.
        self._client.upsert(collection_name=self._collection, points=points, wait=True)

    def search(self, vector: list[float], k: int) -> list[RetrievedChunk]:
        result = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            using=DENSE_VECTOR_NAME,
            limit=k,
            with_payload=True,
        )
        return [
            RetrievedChunk(chunk=DocumentChunk.model_validate(p.payload), score=p.score)
            for p in result.points
        ]
