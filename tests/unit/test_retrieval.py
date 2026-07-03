from nimbusdesk.domain.knowledge import DocumentChunk
from nimbusdesk.rag.chunking import make_chunk_id
from nimbusdesk.rag.retrieval import Retriever
from tests.fakes import FakeEmbedder, InMemoryVectorIndex


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
    index = InMemoryVectorIndex()
    chunks = [_chunk("doc", i, text) for i, text in enumerate(texts)]
    index.upsert(chunks, embedder.embed_passages([c.text for c in chunks]))
    return Retriever(embedder, index)


def test_exact_text_query_ranks_its_chunk_first():
    # FakeEmbedder is hash-based: identical text -> identical vector -> score 1.0.
    # This verifies the query vector actually reaches the index unmangled.
    retriever = _make_retriever(["refund policy", "rate limits", "sso setup"])
    results = retriever.search("rate limits", k=3)

    assert results[0].chunk.text == "rate limits"
    assert results[0].score > 0.999


def test_respects_k_and_returns_sorted_scores():
    retriever = _make_retriever([f"text {i}" for i in range(10)])
    results = retriever.search("text 3", k=4)

    assert len(results) == 4
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
