"""Human-in-the-loop through the real graph: the interrupt fires, the graph
pauses on its checkpoint, and a human decision resumes it — including from a
brand-new graph object, which is the 'operator answers days later in another
process' scenario the whole architecture was chosen for."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from nimbusdesk.agents.graph import build_support_graph, run_support_graph
from nimbusdesk.llm.ports import AssistantTurn, ToolCall
from nimbusdesk.rag.retrieval import Retriever
from tests.fakes import (
    FakeEmbedder,
    FakeLLMProvider,
    FakeReranker,
    FakeSparseEmbedder,
    FakeToolLLM,
    InMemoryVectorIndex,
)

TRIAGE_BILLING = (
    '{"category": "billing", "priority": "normal", '
    '"confidence": 0.9, "summary": "Refund request"}'
)


def _refund_turns(amount: float) -> list[AssistantTurn]:
    return [
        AssistantTurn(
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="request_refund",
                    arguments={
                        "email": "dana@acme.io",
                        "amount_usd": amount,
                        "reason": "duplicate charge",
                    },
                )
            ]
        ),
        AssistantTurn(text="Your refund has been processed."),
    ]


def _graph(strong, checkpointer=None):
    retriever = Retriever(FakeEmbedder(), FakeSparseEmbedder(), InMemoryVectorIndex())
    return build_support_graph(
        FakeLLMProvider([TRIAGE_BILLING]), strong, retriever, FakeReranker(),
        checkpointer=checkpointer or InMemorySaver(),
    )


def test_small_refund_needs_no_human():
    graph = _graph(FakeToolLLM(_refund_turns(120.0)))
    approvals: list[dict] = []

    state = run_support_graph(
        graph, "refund my duplicate $120 charge", "dana@acme.io",
        approval_callback=lambda p: approvals.append(p) or {"approved": True},
    )

    assert approvals == [], "under the limit, no interrupt must fire"
    assert state.resolved_by == "billing"
    assert state.pending_refund is None


def test_large_refund_pauses_for_approval_and_resumes_approved():
    graph = _graph(FakeToolLLM(_refund_turns(800.0)))
    seen_payloads: list[dict] = []

    def approve(payload: dict) -> dict:
        seen_payloads.append(payload)
        return {"approved": True}

    state = run_support_graph(
        graph, "I was double charged $800", "dana@acme.io", approval_callback=approve
    )

    # The human saw exactly what they were approving...
    assert seen_payloads[0]["action"] == "issue_refund"
    assert seen_payloads[0]["amount_usd"] == 800.0
    # ...and their decision, not the model's draft, became the answer.
    assert state.refund_decision == "approved"
    assert state.resolved_by == "billing+human"
    assert "approved" in (state.final_answer or "")


def test_denied_refund_ships_the_denial_with_the_note():
    graph = _graph(FakeToolLLM(_refund_turns(9000.0)))
    state = run_support_graph(
        graph, "refund me $9000", "dana@acme.io",
        approval_callback=lambda p: {"approved": False, "note": "outside refund window"},
    )

    assert state.refund_decision == "denied"
    assert state.escalated
    assert "outside refund window" in (state.final_answer or "")


def test_default_callback_fails_closed():
    # No human available -> the irreversible action is denied, never executed.
    graph = _graph(FakeToolLLM(_refund_turns(800.0)))
    state = run_support_graph(graph, "refund me $800", "dana@acme.io")
    assert state.refund_decision == "denied"


def test_resume_works_from_a_fresh_graph_instance_days_later():
    """The killer property: the pause survives the process. We interrupt with
    one graph object, then build a NEW graph over the same checkpointer and
    resume there — as a different worker would, days later."""
    checkpointer = InMemorySaver()
    graph_a = _graph(FakeToolLLM(_refund_turns(800.0)), checkpointer)

    config = {"configurable": {"thread_id": "ticket-7"}}
    result = graph_a.invoke(
        {"question": "refund my $800", "customer_email": "dana@acme.io",
         "thread_hint": "ticket-7"},
        config=config,
    )
    assert "__interrupt__" in result, "the run must be paused"

    # "Days later": a fresh graph instance (fresh LLM fakes and all) over the
    # SAME checkpointer — only the persisted state connects the two.
    graph_b = _graph(FakeToolLLM([]), checkpointer)
    resumed = graph_b.invoke(Command(resume={"approved": True}), config=config)

    assert resumed["refund_decision"] == "approved"
    assert "approved" in resumed["final_answer"]