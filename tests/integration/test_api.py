"""The HTTP surface, tested with a fake-wired runtime injected into the app.

The SSE contract and the two-call HITL flow (stream pauses -> approvals
endpoint resumes) are exactly what a frontend/operator tool would build
against, so they get pinned here.
"""

import json

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from nimbusdesk.agents.graph import build_support_graph
from nimbusdesk.infrastructure.settings import Settings
from nimbusdesk.interface.api.app import create_app
from nimbusdesk.interface.wiring import AppRuntime
from nimbusdesk.llm.ports import AssistantTurn, ToolCall
from nimbusdesk.llm.tracking import UsageTracker
from nimbusdesk.rag.retrieval import Retriever
from tests.fakes import (
    FakeEmbedder,
    FakeLLMProvider,
    FakeReranker,
    FakeSparseEmbedder,
    FakeToolLLM,
    InMemoryVectorIndex,
)

TRIAGE_TECHNICAL = (
    '{"category": "technical", "priority": "normal", '
    '"confidence": 0.9, "summary": "Sync issue"}'
)
TRIAGE_BILLING = (
    '{"category": "billing", "priority": "normal", '
    '"confidence": 0.9, "summary": "Refund request"}'
)


def _client(fast_responses, strong_turns) -> TestClient:
    fast = UsageTracker(FakeLLMProvider(fast_responses))
    strong = UsageTracker(FakeToolLLM(strong_turns))
    graph = build_support_graph(
        fast, strong,
        Retriever(FakeEmbedder(), FakeSparseEmbedder(), InMemoryVectorIndex()),
        FakeReranker(),
        checkpointer=InMemorySaver(),
    )
    runtime = AppRuntime(graph=graph, fast=fast, strong=strong, settings=Settings())
    return TestClient(create_app(runtime=runtime))


def _sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    event = None
    for line in text.splitlines():
        if line.startswith("event: "):
            event = line.removeprefix("event: ")
        elif line.startswith("data: "):
            events.append((event, json.loads(line.removeprefix("data: "))))
    return events


def test_health_ok():
    client = _client([], [])
    assert client.get("/health").json() == {"status": "ok"}


def test_chat_streams_node_progress_then_answer():
    client = _client(
        [TRIAGE_TECHNICAL],
        [AssistantTurn(text="Sync is degraded; known incident.")],
    )
    response = client.post(
        "/chat", json={"message": "sync is slow", "thread_id": "t1"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _sse_events(response.text)
    node_events = [d["node"] for e, d in events if e == "node"]
    assert node_events[0] == "guard_input"
    assert "triage" in node_events and "technical" in node_events

    final = [d for e, d in events if e == "answer"]
    assert final[0]["answer"] == "Sync is degraded; known incident."
    assert final[0]["resolved_by"] == "technical"
    assert "session_est_cost_usd" in final[0]


def test_hitl_over_http_pause_then_approve():
    """The two-call flow: /chat pauses with approval_required, a later call
    to /approvals/{thread} resumes the SAME checkpointed run."""
    client = _client(
        [TRIAGE_BILLING],
        [
            AssistantTurn(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="request_refund",
                        arguments={
                            "email": "dana@acme.io",
                            "amount_usd": 900.0,
                            "reason": "duplicate charge",
                        },
                    )
                ]
            ),
            AssistantTurn(text="unused draft"),
        ],
    )

    response = client.post(
        "/chat",
        json={"message": "refund my $900", "email": "dana@acme.io", "thread_id": "tk-9"},
    )
    events = _sse_events(response.text)
    pauses = [d for e, d in events if e == "approval_required"]
    assert pauses and pauses[0]["amount_usd"] == 900.0
    assert pauses[0]["thread_id"] == "tk-9"
    assert not [d for e, d in events if e == "answer"], "no answer while paused"

    # The operator's separate HTTP call, later:
    result = client.post("/approvals/tk-9", json={"approved": True}).json()
    assert result["resolved_by"] == "billing+human"
    assert "approved" in result["answer"]


def test_approving_a_thread_with_nothing_pending_is_409():
    client = _client([], [])
    response = client.post("/approvals/ghost-thread", json={"approved": True})
    assert response.status_code == 409


def test_degraded_runtime_returns_503_with_reason(mocker):
    # Startup failures (e.g. missing API key) must degrade the service with a
    # clear reason, never crash-loop the container.
    mocker.patch(
        "nimbusdesk.interface.api.app._build_production_runtime",
        return_value=(None, "MissingApiKeyError: no key"),
    )
    with TestClient(create_app()) as client:
        health = client.get("/health").json()
        assert health["status"] == "degraded"
        assert "MissingApiKeyError" in health["detail"]
        assert client.post("/chat", json={"message": "hi"}).status_code == 503


@pytest.mark.parametrize("bad", [{"message": ""}, {}])
def test_input_schema_is_enforced_at_the_http_boundary(bad):
    client = _client([], [])
    assert client.post("/chat", json=bad).status_code == 422