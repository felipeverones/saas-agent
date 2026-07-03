from nimbusdesk.domain.knowledge import DocumentChunk
from nimbusdesk.rag.chunking import make_chunk_id
from nimbusdesk.rag.retrieval import Retriever
from tests.fakes import FakeEmbedder, FakeSparseEmbedder, InMemoryVectorIndex


def _chunk(doc_id: str, position: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=make_chunk_id(doc_id, position),
        doc_id=doc_id,
        title=doc_id,
        section="S",
        text=text,
        position=position,
    )


def _make_retriever(texts: list[str]) -> Retriever:
    embedder = FakeEmbedder()
    sparse_embedder = FakeSparseEmbedder()
    index = InMemoryVectorIndex()
    chunks = [_chunk("doc", i, text) for i, text in enumerate(texts)]
    index.upsert(
        chunks,
        embedder.embed_passages([c.text for c in chunks]),
        sparse_embedder.embed_passages([c.text for c in chunks]),
    )
    return Retriever(embedder, sparse_embedder, index)


def test_exact_text_query_ranks_its_chunk_first():
    # FakeEmbedder is hash-based: identical text -> identical vectors on BOTH
    # channels -> top rank in both -> top RRF fusion score. This verifies the
    # query vectors actually reach the index unmangled.
    retriever = _make_retriever(["refund policy", "rate limits", "sso setup"])
    results = retriever.search("rate limits", k=3)

    assert results[0].chunk.text == "rate limits"


def test_respects_k_and_returns_sorted_scores():
    retriever = _make_retriever([f"text {i}" for i in range(10)])
    results = retriever.search("text 3", k=4)

    assert len(results) == 4
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
