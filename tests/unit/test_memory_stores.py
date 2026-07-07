"""The two long-term stores, each tested on its own axis:
profile = exact lookup + consolidation; episodic = similarity + isolation."""

import sqlite3

from qdrant_client import QdrantClient

from nimbusdesk.memory.episodic import EpisodicMemoryStore
from nimbusdesk.memory.profile_store import SqliteProfileStore
from tests.fakes import FakeEmbedder


def _profiles() -> SqliteProfileStore:
    return SqliteProfileStore(sqlite3.connect(":memory:"))


def _episodes() -> EpisodicMemoryStore:
    return EpisodicMemoryStore(QdrantClient(":memory:"), FakeEmbedder(), "test_memories")


def test_profile_facts_roundtrip_and_email_isolation():
    store = _profiles()
    store.upsert_facts("dana@acme.io", {"plan": "business", "language": "en"})
    store.upsert_facts("sam@nimbusfan.com", {"plan": "free"})

    assert store.get_profile("dana@acme.io") == {"language": "en", "plan": "business"}
    assert store.get_profile("sam@nimbusfan.com") == {"plan": "free"}
    assert store.get_profile("ghost@nowhere.io") == {}


def test_profile_upsert_consolidates_instead_of_accumulating():
    store = _profiles()
    store.upsert_facts("dana@acme.io", {"plan": "pro"})
    store.upsert_facts("dana@acme.io", {"plan": "business"})  # upgrade happened

    # One key, latest value — no contradictory duplicates.
    assert store.get_profile("dana@acme.io") == {"plan": "business"}


def test_episodes_recall_by_similarity():
    store = _episodes()
    store.store("dana@acme.io", "t1", 0, "sync stuck at 99 percent on Windows")
    store.store("dana@acme.io", "t1", 1, "asked about invoice for May")

    # FakeEmbedder is hash-based: the identical text is the nearest neighbor.
    results = store.recall("dana@acme.io", "sync stuck at 99 percent on Windows", k=1)
    assert results[0].summary.startswith("sync stuck")


def test_episode_recall_is_isolated_per_customer():
    store = _episodes()
    store.store("dana@acme.io", "t1", 0, "enterprise contract negotiation details")

    # Sam must never see Dana's history, no matter how similar the query —
    # the filter is enforced in the database, not in post-processing.
    assert store.recall("sam@nimbusfan.com", "enterprise contract negotiation details") == []


def test_episode_storage_is_idempotent_per_turn():
    store = _episodes()
    store.store("dana@acme.io", "t1", 0, "first version of the summary")
    store.store("dana@acme.io", "t1", 0, "revised version of the summary")

    results = store.recall("dana@acme.io", "summary", k=10)
    assert len(results) == 1, "same (thread, turn) must upsert, not duplicate"
    assert results[0].summary == "revised version of the summary"