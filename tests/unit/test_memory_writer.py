import sqlite3

from qdrant_client import QdrantClient

from nimbusdesk.memory.episodic import EpisodicMemoryStore
from nimbusdesk.memory.profile_store import SqliteProfileStore
from nimbusdesk.memory.service import MemoryService
from nimbusdesk.memory.writer import MemoryWriter
from tests.fakes import ExplodingLLMProvider, FakeEmbedder, FakeLLMProvider

EXTRACTION = (
    '{"summary": "Customer hit sync issues on Windows; advised re-login.", '
    '"facts": {"os": "windows", "plan": "business"}}'
)


def _service(llm) -> tuple[MemoryService, SqliteProfileStore, EpisodicMemoryStore]:
    profiles = SqliteProfileStore(sqlite3.connect(":memory:"))
    episodes = EpisodicMemoryStore(QdrantClient(":memory:"), FakeEmbedder(), "mem")
    service = MemoryService(profiles, episodes, MemoryWriter(llm, profiles, episodes))
    return service, profiles, episodes


def test_extraction_writes_episode_and_facts():
    service, profiles, episodes = _service(FakeLLMProvider([EXTRACTION]))
    service.record_turn("dana@acme.io", "t1", 0, "sync broken on my laptop", "try re-login")

    assert profiles.get_profile("dana@acme.io") == {"os": "windows", "plan": "business"}
    recalled = episodes.recall("dana@acme.io", "sync issues", k=1)
    assert "sync issues on Windows" in recalled[0].summary


def test_unparseable_extraction_still_stores_fallback_episode():
    service, profiles, episodes = _service(FakeLLMProvider(["hmm, interesting exchange"]))
    service.record_turn("dana@acme.io", "t1", 0, "sync broken", "answer")

    assert profiles.get_profile("dana@acme.io") == {}, "no facts from garbage output"
    recalled = episodes.recall("dana@acme.io", "sync broken", k=1)
    assert "Customer asked: sync broken" in recalled[0].summary


def test_llm_outage_fails_open_with_fallback_episode():
    service, _, episodes = _service(ExplodingLLMProvider())
    service.record_turn("dana@acme.io", "t1", 0, "question here", "answer")

    assert episodes.recall("dana@acme.io", "question here", k=1), "memory still recorded"


def test_recall_formats_profile_and_episodes():
    service, _, _ = _service(FakeLLMProvider([EXTRACTION]))
    service.record_turn("dana@acme.io", "t1", 0, "sync broken on my laptop", "re-login")

    context = service.recall("dana@acme.io", "sync problems again")
    assert context is not None
    assert "os: windows" in context
    assert "sync issues on Windows" in context


def test_recall_returns_none_when_nothing_known():
    service, _, _ = _service(FakeLLMProvider([]))
    assert service.recall("ghost@nowhere.io", "anything") is None