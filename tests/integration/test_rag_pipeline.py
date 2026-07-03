"""Integration: real embeddings + real Qdrant engine (in-process) + real corpus.

Unit tests proved the MECHANICS with fakes; these prove retrieval QUALITY —
that a support question phrased in the user's words actually surfaces the
right article. `QdrantClient(":memory:")` runs the full Qdrant logic
in-process, so no Docker is needed.

First run downloads the embedding model (~65 MB, then cached). Skip these
with: pytest -m "not integration".
"""

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from nimbusdesk.infrastructure.embeddings import FastEmbedEmbedder
from nimbusdesk.infrastructure.vector_store import QdrantVectorIndex
from nimbusdesk.rag.ingestion import IngestionPipeline
from nimbusdesk.rag.retrieval import Retriever

pytestmark = pytest.mark.integration

SEED_DIR = Path(__file__).parents[2] / "data" / "seed"


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    embedder = FastEmbedEmbedder(model_name="BAAI/bge-small-en-v1.5", dimension=384)
    index = QdrantVectorIndex(
        client=QdrantClient(":memory:"), collection="test_kb", dimension=384
    )
    report = IngestionPipeline(embedder, index).run(SEED_DIR)
    assert report.chunks > 20, "seed corpus unexpectedly small — did data/seed move?"
    return Retriever(embedder, index)


@pytest.mark.parametrize(
    ("question", "expected_doc"),
    [
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
    ],
)
def test_user_questions_surface_the_right_article(retriever, question, expected_doc):
    results = retriever.search(question, k=3)
    found = [r.chunk.doc_id for r in results]
    assert expected_doc in found, f"expected {expected_doc} in top-3, got {found}"


def test_results_carry_citation_provenance(retriever):
    result = retriever.search("What is the uptime SLA?", k=1)[0]
    # Non-negotiable requirement: every retrieved chunk must be citable —
    # document, section and verbatim text all present.
    assert result.chunk.doc_id
    assert result.chunk.section
    assert result.chunk.text.strip()
