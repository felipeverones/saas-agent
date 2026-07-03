from nimbusdesk.rag.rewrite import QueryRewriter
from tests.fakes import ExplodingLLMProvider, FakeLLMProvider


def test_uses_llm_rewrite():
    llm = FakeLLMProvider(['"enforce two-factor authentication workspace"'])
    result = QueryRewriter(llm).rewrite("my boss wants everyone on 2fa, can we force it?")
    assert result == "enforce two-factor authentication workspace"


def test_llm_outage_falls_back_to_raw_question():
    # Graceful degradation: the rewrite is an optimization, not a dependency.
    question = "how do refunds work?"
    assert QueryRewriter(ExplodingLLMProvider()).rewrite(question) == question


def test_degenerate_rewrites_are_rejected():
    question = "how do refunds work?"
    assert QueryRewriter(FakeLLMProvider([""])).rewrite(question) == question
    assert QueryRewriter(FakeLLMProvider(["word " * 200])).rewrite(question) == question
