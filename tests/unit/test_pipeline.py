"""End-to-end pipeline behavior against fakes — especially the bounded
self-correction loop, which is the "agentic" part worth pinning with tests."""

from nimbusdesk.llm.tracking import UsageTracker
from nimbusdesk.rag.answering import AnswerGenerator
from nimbusdesk.rag.ingestion import IngestionPipeline
from nimbusdesk.rag.pipeline import GroundedRagPipeline
from nimbusdesk.rag.retrieval import Retriever
from nimbusdesk.rag.rewrite import QueryRewriter
from nimbusdesk.rag.self_check import FaithfulnessChecker
from tests.fakes import (
    FakeEmbedder,
    FakeLLMProvider,
    FakeReranker,
    FakeSparseEmbedder,
    InMemoryVectorIndex,
)
from tests.unit.test_ingestion import _write_corpus


def _pipeline(tmp_path, fast_responses: list[str], strong_responses: list[str]):
    index = InMemoryVectorIndex()
    dense, sparse = FakeEmbedder(), FakeSparseEmbedder()
    IngestionPipeline(dense, sparse, index).run(_write_corpus(tmp_path))
    fast = UsageTracker(FakeLLMProvider(fast_responses))
    strong_fake = FakeLLMProvider(strong_responses)
    strong = UsageTracker(strong_fake)
    return (
        GroundedRagPipeline(
            rewriter=QueryRewriter(fast),
            retriever=Retriever(dense, sparse, index),
            reranker=FakeReranker(),
            generator=AnswerGenerator(strong),
            checker=FaithfulnessChecker(fast),
            usage_trackers=[fast, strong],
            top_k=2,
        ),
        strong_fake,
    )


def test_happy_path_is_grounded_with_usage(tmp_path):
    pipeline, _ = _pipeline(
        tmp_path,
        fast_responses=["alpha text search", '{"unsupported_claims": []}'],
        strong_responses=["Alpha is documented [1]."],
    )
    result = pipeline.ask("what is alpha?")

    assert result.grounded
    assert [c.marker for c in result.citations] == [1]
    # 2 fast calls + 1 strong call, 10 in / 5 out each — usage must aggregate.
    assert result.input_tokens == 30 and result.output_tokens == 15


def test_failed_check_triggers_exactly_one_retry(tmp_path):
    pipeline, strong = _pipeline(
        tmp_path,
        fast_responses=[
            "alpha text search",
            '{"unsupported_claims": ["alpha costs $99"]}',  # first check: fail
            '{"unsupported_claims": []}',                   # second check: pass
        ],
        strong_responses=["Alpha costs $99 [1].", "Alpha is documented [1]."],
    )
    result = pipeline.ask("what is alpha?")

    assert result.grounded
    assert result.answer == "Alpha is documented [1]."
    assert len(strong.calls) == 2, "exactly one corrective regeneration"
    # The revision note must reach the second generation prompt.
    assert "alpha costs $99" in strong.calls[1]["messages"][0].content


def test_persistent_failure_ships_flagged_not_looping(tmp_path):
    pipeline, strong = _pipeline(
        tmp_path,
        fast_responses=[
            "alpha text search",
            '{"unsupported_claims": ["bad claim"]}',
            '{"unsupported_claims": ["bad claim"]}',  # still failing after retry
        ],
        strong_responses=["Bad answer [1].", "Still bad [1]."],
    )
    result = pipeline.ask("what is alpha?")

    assert not result.grounded
    assert "bad claim" in (result.notes or "")
    assert len(strong.calls) == 2, "the correction loop must be bounded"


def test_inconclusive_check_ships_with_note(tmp_path):
    pipeline, _ = _pipeline(
        tmp_path,
        fast_responses=["alpha text search", "not json at all"],
        strong_responses=["Alpha is documented [1]."],
    )
    result = pipeline.ask("what is alpha?")

    assert result.grounded
    assert "unverified" in (result.notes or "")
