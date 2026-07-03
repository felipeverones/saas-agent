"""Test doubles for the RAG and LLM ports.

THE PAYOFF OF PORTS & ADAPTERS
These fakes satisfy the same Protocols as FastEmbed/Qdrant/Anthropic
(structural typing: no imports from the real adapters needed), so every
pipeline test runs in microseconds with zero network, zero model downloads,
zero token cost.

FakeEmbedder / FakeSparseEmbedder are deterministic, NOT semantic: identical
texts get identical vectors. Enough to test MECHANICS (routing, ranking,
idempotency); semantic quality belongs to the integration suite.
"""

import hashlib
import math
import re
from typing import Sequence

from nimbusdesk.domain.knowledge import DocumentChunk, RetrievedChunk
from nimbusdesk.llm.ports import Completion, Message
from nimbusdesk.rag.ports import SparseVector


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


class FakeSparseEmbedder:
    """Token counts as term weights — a truthful miniature of BM25 mechanics."""

    def _sparse(self, text: str) -> SparseVector:
        weights: dict[int, float] = {}
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            index = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)
            weights[index] = weights.get(index, 0.0) + 1.0
        return SparseVector(indices=list(weights.keys()), values=list(weights.values()))

    def embed_passages(self, texts: Sequence[str]) -> list[SparseVector]:
        return [self._sparse(text) for text in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self._sparse(text)


class InMemoryVectorIndex:
    """Dict-backed hybrid VectorIndex: cosine + sparse-dot ranks fused via RRF —
    the same fusion idea the real Qdrant adapter delegates to the server."""

    def __init__(self) -> None:
        self._points: dict[str, tuple[DocumentChunk, list[float], SparseVector]] = {}

    def __len__(self) -> int:
        return len(self._points)

    def ensure_ready(self) -> None:
        pass

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        dense: Sequence[list[float]],
        sparse: Sequence[SparseVector],
    ) -> None:
        for chunk, dense_vec, sparse_vec in zip(chunks, dense, sparse, strict=True):
            self._points[chunk.chunk_id] = (chunk, dense_vec, sparse_vec)

    def search(self, dense: list[float], sparse: SparseVector, k: int) -> list[RetrievedChunk]:
        entries = list(self._points.values())
        dense_rank = _ranks([_cosine(dense, d) for _, d, _ in entries])
        sparse_rank = _ranks([_sparse_dot(sparse, s) for _, _, s in entries])
        fused = [
            (1 / (60 + dense_rank[i]) + 1 / (60 + sparse_rank[i]), entries[i][0])
            for i in range(len(entries))
        ]
        fused.sort(key=lambda pair: pair[0], reverse=True)
        return [RetrievedChunk(chunk=chunk, score=score) for score, chunk in fused[:k]]


class FakeReranker:
    """Pass-through reranker (truncates to top_n) — pipeline tests don't care
    about ordering quality, only that the stage is invoked and bounded."""

    def rerank(
        self, query: str, results: Sequence[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        return list(results[:top_n])


class FakeLLMProvider:
    """Returns scripted responses in order and records every request, so tests
    can assert BOTH what the pipeline sent and how it used the reply."""

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(
        self,
        *,
        messages: Sequence[Message],
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        self.calls.append({"system": system, "messages": list(messages)})
        if not self._responses:
            raise AssertionError("FakeLLMProvider ran out of scripted responses")
        return Completion(
            text=self._responses.pop(0), model="fake", input_tokens=10, output_tokens=5
        )


class ExplodingLLMProvider:
    """Always raises — for testing graceful-degradation paths."""

    def complete(self, **kwargs) -> Completion:
        raise ConnectionError("simulated LLM outage")


def _ranks(scores: list[float]) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranks = [0] * len(scores)
    for rank, index in enumerate(order):
        ranks[index] = rank
    return ranks


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (norm_a * norm_b)


def _sparse_dot(a: SparseVector, b: SparseVector) -> float:
    weights = dict(zip(a.indices, a.values, strict=True))
    return sum(weights.get(i, 0.0) * v for i, v in zip(b.indices, b.values, strict=True))
