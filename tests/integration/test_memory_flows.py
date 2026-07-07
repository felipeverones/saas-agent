"""Memory through the REAL graph: multi-turn short-term history on one thread,
and long-term recall across DIFFERENT threads (the 'customer returns next
week and we remember them' scenario)."""

import sqlite3

from langgraph.checkpoint.memory import InMemorySaver
from qdrant_client import QdrantClient

from nimbusdesk.agents.graph import build_support_graph, run_support_graph
from nimbusdesk.llm.ports import AssistantTurn, UserTurn
from nimbusdesk.memory.episodic import EpisodicMemoryStore
from nimbusdesk.memory.profile_store import SqliteProfileStore
from nimbusdesk.memory.service import MemoryService
from nimbusdesk.memory.writer import MemoryWriter
from nimbusdesk.rag.retrieval import Retriever
from tests.fakes import (
    FakeEmbedder,
    FakeLLMProvider,
    FakeReranker,
    FakeSparseEmbedder,
    FakeToolLLM,
    InMemoryVectorIndex,
)

TRIAGE = (
    '{"category": "technical", "priority": "normal", '
    '"confidence": 0.9, "summary": "Sync problem"}'
)
EXTRACTION = (
    '{"summary": "Customer had sync stuck on a Windows laptop; advised re-login.", '
    '"facts": {"os": "windows"}}'
)


def _memory() -> tuple[SqliteProfileStore, EpisodicMemoryStore]:
    profiles = SqliteProfileStore(sqlite3.connect(":memory:"))
    episodes = EpisodicMemoryStore(QdrantClient(":memory:"), FakeEmbedder(), "mem")
    return profiles, episodes


def _graph(fast, strong, memory: MemoryService):
    retriever = Retriever(FakeEmbedder(), FakeSparseEmbedder(), InMemoryVectorIndex())
    return build_support_graph(
        fast, strong, retriever, FakeReranker(),
        checkpointer=InMemorySaver(), memory=memory,
    )


def test_short_term_history_accumulates_across_turns_on_one_thread():
    profiles, episodes = _memory()
    memory = MemoryService(
        profiles, episodes,
        MemoryWriter(FakeLLMProvider([EXTRACTION, EXTRACTION]), profiles, episodes),
    )
    fast = FakeLLMProvider([TRIAGE, TRIAGE])
    strong = FakeToolLLM(
        [AssistantTurn(text="Try re-logging in."), AssistantTurn(text="Then update the client.")]
    )
    graph = _graph(fast, strong, memory)

    first = run_support_graph(graph, "sync is stuck", "dana@acme.io", thread_id="t1")
    second = run_support_graph(graph, "still stuck after that", "dana@acme.io", thread_id="t1")

    # Short-term memory: the thread's history grew across invocations...
    assert first.turn_index == 1 and second.turn_index == 2
    assert [t.content for t in second.history] == [
        "sync is stuck",
        "Try re-logging in.",
        "still stuck after that",
        "Then update the client.",
    ]
    # ...and the specialist SAW it on turn 2 (the whole point of keeping it).
    second_turn_prompt = next(
        t.content for t in strong.calls[1]["turns"] if isinstance(t, UserTurn)
    )
    assert "Conversation so far" in second_turn_prompt
    assert "Try re-logging in." in second_turn_prompt
    # Per-turn fields were reset between invocations (no stale answer leaked).
    assert second.final_answer == "Then update the client."


def test_long_term_memory_survives_into_a_new_session():
    """Session A (thread a): customer reports a problem; memory is written.
    Session B (thread b, days later): recall injects what we learned."""
    profiles, episodes = _memory()
    memory = MemoryService(
        profiles, episodes,
        MemoryWriter(FakeLLMProvider([EXTRACTION, EXTRACTION]), profiles, episodes),
    )

    # Session A
    graph_a = _graph(
        FakeLLMProvider([TRIAGE]),
        FakeToolLLM([AssistantTurn(text="Advised re-login.")]),
        memory,
    )
    run_support_graph(graph_a, "sync stuck on my windows laptop", "dana@acme.io", thread_id="a")

    # Session B: fresh graph, fresh thread — only the MemoryService is shared,
    # exactly like a new process days later hitting the same stores.
    strong_b = FakeToolLLM([AssistantTurn(text="Welcome back — is this the same laptop?")])
    graph_b = _graph(FakeLLMProvider([TRIAGE]), strong_b, memory)
    state_b = run_support_graph(graph_b, "sync problems again", "dana@acme.io", thread_id="b")

    assert state_b.memory_context is not None
    assert "os: windows" in state_b.memory_context
    assert "sync stuck on a Windows laptop" in state_b.memory_context
    # And it reached the specialist's prompt.
    prompt_b = next(t.content for t in strong_b.calls[0]["turns"] if isinstance(t, UserTurn))
    assert "What we know about dana@acme.io" in prompt_b
    # Short-term history did NOT leak across threads.
    assert state_b.history[0].content == "sync problems again"


def test_memory_of_one_customer_never_leaks_to_another():
    profiles, episodes = _memory()
    memory = MemoryService(
        profiles, episodes,
        MemoryWriter(FakeLLMProvider([EXTRACTION]), profiles, episodes),
    )
    graph_a = _graph(
        FakeLLMProvider([TRIAGE]), FakeToolLLM([AssistantTurn(text="ok")]), memory
    )
    run_support_graph(graph_a, "sync stuck on windows", "dana@acme.io", thread_id="a")

    graph_b = _graph(
        FakeLLMProvider([TRIAGE]), FakeToolLLM([AssistantTurn(text="hi")]), memory
    )
    state = run_support_graph(graph_b, "sync stuck on windows", "sam@nimbusfan.com", "b")

    assert state.memory_context is None, "Dana's memories must be invisible to Sam"