"""Integration: real embeddings + real Qdrant engine (in-process) + real corpus.

Unit tests proved the MECHANICS with fakes; these prove retrieval QUALITY —
that a support question phrased in the user's words actually surfaces the
right article. `QdrantClient(":memory:")` runs the full Qdrant logic
in-process (including hybrid RRF fusion), so no Docker is needed.

Generation/self-check are NOT tested here: they'd need a real LLM and real
money. Retrieval quality is testable for free; answer quality is what the
golden-dataset evals measure in phase 8.

First run downloads the embedding + reranker models (~90 MB total, cached).
Skip with: pytest -m "not integration".
"""

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from nimbusdesk.infrastructure.embeddings import FastEmbedEmbedder, FastEmbedSparseEmbedder
from nimbusdesk.infrastructure.reranker import FastEmbedReranker
from nimbusdesk.infrastructure.vector_store import QdrantVectorIndex
from nimbusdesk.rag.ingestion import IngestionPipeline
from nimbusdesk.rag.retrieval import Retriever

pytestmark = pytest.mark.integration

SEED_DIR = Path(__file__).parents[2] / "data" / "seed"


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    dense = FastEmbedEmbedder(model_name="BAAI/bge-small-en-v1.5", dimension=384)
    sparse = FastEmbedSparseEmbedder(model_name="Qdrant/bm25")
    index = QdrantVectorIndex(
        client=QdrantClient(":memory:"), collection="test_kb", dimension=384
    )
    report = IngestionPipeline(dense, sparse, index).run(SEED_DIR)
    assert report.chunks > 20, "seed corpus unexpectedly small — did data/seed move?"
    return Retriever(dense, sparse, index)


@pytest.mark.parametrize(
    ("question", "expected_doc"),
    [
        # Semantic phrasings — the DENSE channel's home turf.
        (
            "How long do I have to request a refund on an annual subscription?",
            "billing-and-refunds",
        ),
        (
            "SAML login fails with an invalid signature error after certificate rotation",
            "sso-saml",
        ),
        ("The API keeps returning 429 errors, what should my client do?", "api-rate-limits"),
        ("Sync has been stuck at 99 percent on Windows for hours", "sync-troubleshooting"),
        # Exact identifiers — the SPARSE (BM25) channel's home turf. These are
        # precisely the queries dense-only retrieval fumbles (phase 1 -> 2).
        ("error ND-WH-TLS", "webhooks"),
        ("what does ND-SYNC-PATH mean", "sync-troubleshooting"),
    ],
)
def test_user_questions_surface_the_right_article(retriever, question, expected_doc):
    results = retriever.search(question, k=3)
    found = [r.chunk.doc_id for r in results]
    assert expected_doc in found, f"expected {expected_doc} in top-3, got {found}"


def test_reranker_puts_the_answering_chunk_first(retriever):
    """The funnel end to end: 20 hybrid candidates -> cross-encoder -> top-3.
    The chunk that literally contains the answer must win rank 1."""
    reranker = FastEmbedReranker("Xenova/ms-marco-MiniLM-L-6-v2")
    query = "how many days until deleted files are purged from trash?"

    candidates = retriever.search(query, k=20)
    top = reranker.rerank(query, candidates, top_n=3)

    assert top[0].chunk.doc_id == "data-export-retention"
    assert "trash" in top[0].chunk.text.lower()


def test_results_carry_citation_provenance(retriever):
    result = retriever.search("What is the uptime SLA?", k=1)[0]
    # Non-negotiable requirement: every retrieved chunk must be citable —
    # document, section and verbatim text all present.
    assert result.chunk.doc_id
    assert result.chunk.section
    assert result.chunk.text.strip()
