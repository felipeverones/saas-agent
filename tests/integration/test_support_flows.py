"""End-to-end flows through the REAL LangGraph graph — LLMs scripted, graph
wiring, state merging and checkpointing genuine. These are the two complete
flows required by the project spec (resolved ticket / escalated ticket), plus
failure containment and checkpoint persistence.

No pytest marker: fakes make these fast enough for the default suite; the
`integration` marker stays reserved for tests that download embedding models.
"""

from langgraph.checkpoint.memory import InMemorySaver

from nimbusdesk.agents.graph import build_support_graph, run_support_graph
from nimbusdesk.llm.ports import AssistantTurn
from nimbusdesk.rag.retrieval import Retriever
from tests.fakes import (
    ExplodingLLMProvider,
    FakeEmbedder,
    FakeLLMProvider,
    FakeReranker,
    FakeSparseEmbedder,
    FakeToolLLM,
    InMemoryVectorIndex,
)

TRIAGE_TECHNICAL = (
    '{"category": "technical", "priority": "normal", '
    '"confidence": 0.92, "summary": "Sync is slow"}'
)
TRIAGE_BILLING = (
    '{"category": "billing", "priority": "normal", '
    '"confidence": 0.88, "summary": "Refund question"}'
)
TRIAGE_AMBIGUOUS = (
    '{"category": "technical", "priority": "normal", '
    '"confidence": 0.2, "summary": "Unclear request"}'
)


def _graph(fast, strong, checkpointer=None):
    retriever = Retriever(FakeEmbedder(), FakeSparseEmbedder(), InMemoryVectorIndex())
    return build_support_graph(fast, strong, retriever, FakeReranker(), checkpointer)


def test_technical_ticket_resolved_by_technical_specialist():
    graph = _graph(
        fast=FakeLLMProvider([TRIAGE_TECHNICAL]),
        strong=FakeToolLLM([AssistantTurn(text="Sync is degraded; here's the workaround.")]),
    )
    state = run_support_graph(graph, "sync has been slow for an hour")

    assert state.resolved_by == "technical"
    assert state.final_answer == "Sync is degraded; here's the workaround."
    assert not state.escalated and not state.failures
    assert state.triage and state.triage.category == "technical"


def test_billing_ticket_routes_to_billing_specialist():
    graph = _graph(
        fast=FakeLLMProvider([TRIAGE_BILLING]),
        strong=FakeToolLLM([AssistantTurn(text="Annual plans are refundable within 30 days.")]),
    )
    state = run_support_graph(graph, "can I get a refund on my annual plan?")

    assert state.resolved_by == "billing"
    assert not state.escalated


def test_ambiguous_ticket_escalates_to_human():
    # Low-confidence triage: the strong LLM must never even be called.
    strong = FakeToolLLM([])  # would raise if consulted
    graph = _graph(fast=FakeLLMProvider([TRIAGE_AMBIGUOUS]), strong=strong)
    state = run_support_graph(graph, "everything is weird, please help")

    assert state.escalated and state.resolved_by == "escalation"
    assert "low-confidence triage" in (state.escalation_reason or "")
    assert strong.calls == []


def test_specialist_crash_is_contained_and_escalated():
    """One agent failing must not derrubar the flow: the crash becomes a
    recorded failure and the ticket lands with a human, with an answer."""
    graph = _graph(fast=FakeLLMProvider([TRIAGE_TECHNICAL]), strong=ExplodingLLMProvider())
    state = run_support_graph(graph, "sync is broken")

    assert state.escalated
    assert state.failures and "ConnectionError" in state.failures[0]
    assert state.final_answer, "even a failed run must end with a customer-facing answer"


def test_checkpointing_persists_state_per_thread():
    checkpointer = InMemorySaver()
    graph = _graph(
        fast=FakeLLMProvider([TRIAGE_TECHNICAL]),
        strong=FakeToolLLM([AssistantTurn(text="answer")]),
        checkpointer=checkpointer,
    )
    run_support_graph(graph, "sync slow", thread_id="ticket-42")

    # The state snapshot survives the run, keyed by thread — this is what
    # phase 7 resumes from after a human approval, and phase 6 builds on.
    snapshot = graph.get_state({"configurable": {"thread_id": "ticket-42"}})
    assert snapshot.values["final_answer"] == "answer"

    other = graph.get_state({"configurable": {"thread_id": "ticket-99"}})
    assert not other.values, "threads must be isolated from each other"