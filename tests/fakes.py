"""Test doubles for the RAG ports.

THE PAYOFF OF PORTS & ADAPTERS
These fakes satisfy the same Protocols as FastEmbed/Qdrant (structural typing:
no imports from the real adapters needed), so every pipeline test runs in
microseconds with zero network, zero model downloads, zero cost.

FakeEmbedder is deterministic (hash-based), NOT semantic: identical texts get
identical vectors, different texts get different ones. That's enough to test
MECHANICS (routing, ranking, idempotency). Semantic quality is tested in the
integration suite with the real model.
"""

import hashlib
import math
from typing import Sequence

from nimbusdesk.domain.knowledge import DocumentChunk, RetrievedChunk


class FakeEmbedder:
    dimension = 32

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [byte / 255.0 for byte in digest[: self.dimension]]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class InMemoryVectorIndex:
    """Dict-backed VectorIndex with brute-force cosine search."""

    def __init__(self) -> None:
        self._points: dict[str, tuple[DocumentChunk, list[float]]] = {}

    def __len__(self) -> int:
        return len(self._points)

    def ensure_ready(self) -> None:
        pass

    def upsert(self, chunks: Sequence[DocumentChunk], vectors: Sequence[list[float]]) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._points[chunk.chunk_id] = (chunk, vector)

    def search(self, vector: list[float], k: int) -> list[RetrievedChunk]:
        scored = [
            (_cosine(vector, stored_vector), chunk)
            for chunk, stored_vector in self._points.values()
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [RetrievedChunk(chunk=chunk, score=score) for score, chunk in scored[:k]]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (norm_a * norm_b)
