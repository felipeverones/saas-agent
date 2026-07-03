from nimbusdesk.domain.knowledge import DocumentChunk, RetrievedChunk
from nimbusdesk.rag.answering import AnswerGenerator, extract_citations
from tests.fakes import FakeLLMProvider


def _chunks(n: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=DocumentChunk(
                chunk_id=f"id-{i}",
                doc_id=f"doc-{i}",
                title=f"Title {i}",
                section=f"Section {i}",
                text=f"Fact number {i}.",
                position=i,
            ),
            score=1.0 - i / 10,
        )
        for i in range(1, n + 1)
    ]


def test_citations_extracted_from_markers():
    citations = extract_citations("Refunds take 30 days [1]. Fees differ [3].", _chunks(3))
    assert [c.marker for c in citations] == [1, 3]
    assert citations[0].doc_id == "doc-1"
    assert citations[1].section == "Section 3"


def test_hallucinated_markers_are_ignored():
    # The model cited [7] with only 2 documents: refuse to fabricate provenance.
    citations = extract_citations("A claim [7] and a real one [2].", _chunks(2))
    assert [c.marker for c in citations] == [2]


def test_generator_sends_context_and_parses_answer():
    llm = FakeLLMProvider(["Refunds are available within 30 days [1]."])
    draft = AnswerGenerator(llm).generate("refund window?", _chunks(2))

    sent = llm.calls[0]["messages"][0].content
    assert '<document index="1"' in sent and '<document index="2"' in sent
    assert "refund window?" in sent
    assert draft.citations[0].marker == 1


def test_revision_note_reaches_the_prompt():
    llm = FakeLLMProvider(["Corrected answer [1]."])
    AnswerGenerator(llm).generate(
        "q?", _chunks(1), revision_note="refunds take 90 days"
    )
    assert "refunds take 90 days" in llm.calls[0]["messages"][0].content
