from nimbusdesk.domain.knowledge import DocumentChunk, RetrievedChunk
from nimbusdesk.rag.self_check import FaithfulnessChecker
from tests.fakes import ExplodingLLMProvider, FakeLLMProvider

CHUNKS = [
    RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id="c1", doc_id="d1", title="T", section="S",
            text="Refunds within 30 days.", position=0,
        ),
        score=1.0,
    )
]


def test_grounded_when_no_unsupported_claims():
    checker = FaithfulnessChecker(FakeLLMProvider(['{"unsupported_claims": []}']))
    result = checker.check("Refunds within 30 days [1].", CHUNKS)
    assert result.grounded and not result.inconclusive


def test_unsupported_claims_flagged():
    checker = FaithfulnessChecker(
        FakeLLMProvider(['{"unsupported_claims": ["refunds take 90 days"]}'])
    )
    result = checker.check("Refunds take 90 days [1].", CHUNKS)
    assert not result.grounded
    assert result.unsupported_claims == ["refunds take 90 days"]


def test_json_wrapped_in_prose_still_parses():
    # LLMs love adding "Here is the JSON:" — the parser must tolerate it.
    checker = FaithfulnessChecker(
        FakeLLMProvider(['Here is my audit:\n{"unsupported_claims": []}\nDone.'])
    )
    assert checker.check("answer", CHUNKS).grounded


def test_unparseable_reply_fails_open_as_inconclusive():
    checker = FaithfulnessChecker(FakeLLMProvider(["I think it looks fine?"]))
    result = checker.check("answer", CHUNKS)
    assert result.grounded and result.inconclusive


def test_llm_outage_fails_open_as_inconclusive():
    result = FaithfulnessChecker(ExplodingLLMProvider()).check("answer", CHUNKS)
    assert result.grounded and result.inconclusive
