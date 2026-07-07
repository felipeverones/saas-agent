"""Episodic memory — the SIMILARITY-LOOKUP half of long-term memory.

Past interactions ("had a sync-stuck issue in March, fixed by re-login") are
retrieved by RELEVANCE to the current question, not by exact key — you don't
know in advance which past episode will matter. That's a vector search over
episode summaries: the same machinery as RAG, pointed at our own history
instead of at documentation.

TWO DELIBERATE DIFFERENCES FROM THE RAG INDEX
- Dense vectors only, no BM25 channel: recall queries are paraphrases of past
  situations, not error-code lookups; hybrid would add moving parts for
  little gain here. (If episodes started carrying codes, we'd revisit.)
- A HARD per-customer filter on every search: memory isolation is a security
  property, not a relevance heuristic. Dana's history must be unreachable
  from a conversation with Sam even if it's semantically similar — filtering
  happens in the database, not in post-processing we could forget.
"""

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel
from qdrant_client import QdrantClient, models

from nimbusdesk.rag.ports import Embedder


class Episode(BaseModel):
    episode_id: str
    email: str
    thread_id: str
    summary: str
    created_at: str
    score: float = 0.0


class EpisodicMemoryStore:
    def __init__(self, client: QdrantClient, embedder: Embedder, collection: str) -> None:
        self._client = client
        self._embedder = embedder
        self._collection = collection

    def ensure_ready(self) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=self._embedder.dimension, distance=models.Distance.COSINE
            ),
        )

    def store(self, email: str, thread_id: str, turn_index: int, summary: str) -> None:
        # Deterministic id per (thread, turn): re-processing a turn upserts
        # instead of duplicating — same idempotency recipe as RAG ingestion.
        episode_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"nimbusdesk:memory:{thread_id}:{turn_index}")
        )
        self.ensure_ready()
        self._client.upsert(
            collection_name=self._collection,
            points=[
                models.PointStruct(
                    id=episode_id,
                    vector=self._embedder.embed_passages([summary])[0],
                    payload={
                        "email": email.lower(),
                        "thread_id": thread_id,
                        "summary": summary,
                        "created_at": datetime.now(UTC).isoformat(),
                    },
                )
            ],
            wait=True,
        )

    def recall(self, email: str, query: str, k: int = 3) -> list[Episode]:
        self.ensure_ready()
        result = self._client.query_points(
            collection_name=self._collection,
            query=self._embedder.embed_query(query),
            # The isolation filter: enforced by the DB on every query.
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="email", match=models.MatchValue(value=email.lower())
                    )
                ]
            ),
            limit=k,
            with_payload=True,
        )
        return [
            Episode(
                episode_id=str(p.id),
                email=p.payload["email"],
                thread_id=p.payload["thread_id"],
                summary=p.payload["summary"],
                created_at=p.payload["created_at"],
                score=p.score,
            )
            for p in result.points
        ]
