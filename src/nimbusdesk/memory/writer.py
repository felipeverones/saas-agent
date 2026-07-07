"""Memory writer — the EXTRACT step of extract -> consolidate -> retrieve.

After each completed turn, a cheap LLM distills the exchange into:
- a one-sentence EPISODE summary (goes to the episodic vector store), and
- zero or more durable FACTS about the customer (go to the profile store,
  where upsert-by-key consolidates them).

WHY DISTILL INSTEAD OF STORING THE RAW TRANSCRIPT
Long-term memory is a lossy, searchable digest — that's the asymmetry with
short-term memory (an exact snapshot restored by thread id). Raw transcripts
bloat storage, retrieve poorly (noise dilutes similarity), and turn every
future recall into a re-reading job. Distill once at write time; read cheaply
forever.

FAILURE POLICY: memory is an enhancement, never a dependency — every failure
fails OPEN. If extraction breaks, we store a truncated raw summary and no
facts; if storage breaks, we log and move on. A support answer must never
fail because the diary was unavailable.
"""

import logging

from nimbusdesk.llm.json_parsing import extract_json_object
from nimbusdesk.llm.ports import LLMProvider, Message
from nimbusdesk.memory.episodic import EpisodicMemoryStore
from nimbusdesk.memory.profile_store import SqliteProfileStore

logger = logging.getLogger(__name__)

_SYSTEM = """You maintain long-term memory for a customer support system.

Given one support exchange, reply with ONLY this JSON:
{"summary": "<one sentence: what the customer needed and the outcome>",
 "facts": {"<snake_case_key>": "<value>", ...}}

Facts must be DURABLE properties of the customer worth remembering across
future conversations (plan, role, preferences, environment, recurring
issues). Do NOT record one-off details, dates, or anything speculative.
Return "facts": {} when nothing durable was revealed."""

_MAX_FALLBACK_SUMMARY = 200


class MemoryWriter:
    def __init__(
        self,
        llm: LLMProvider,
        profile_store: SqliteProfileStore,
        episodic_store: EpisodicMemoryStore,
    ) -> None:
        self._llm = llm
        self._profiles = profile_store
        self._episodes = episodic_store

    def record_turn(
        self, email: str, thread_id: str, turn_index: int, question: str, answer: str
    ) -> None:
        summary, facts = self._extract(question, answer)
        try:
            self._episodes.store(email, thread_id, turn_index, summary)
            if facts:
                self._profiles.upsert_facts(email, facts)
        except Exception as error:
            logger.warning(
                "memory write failed (%s: %s); continuing without it",
                type(error).__name__,
                error,
            )

    def _extract(self, question: str, answer: str) -> tuple[str, dict[str, str]]:
        fallback = f"Customer asked: {question}"[:_MAX_FALLBACK_SUMMARY]
        try:
            completion = self._llm.complete(
                system=_SYSTEM,
                messages=[
                    Message(
                        role="user",
                        content=f"Customer: {question}\n\nAssistant: {answer}",
                    )
                ],
                max_tokens=300,
            )
        except Exception as error:
            logger.warning("memory extraction failed (%s: %s)", type(error).__name__, error)
            return fallback, {}

        data = extract_json_object(completion.text)
        if data is None or not isinstance(data.get("summary"), str) or not data["summary"]:
            return fallback, {}
        facts = data.get("facts")
        clean_facts = (
            {str(k): str(v) for k, v in facts.items()} if isinstance(facts, dict) else {}
        )
        return data["summary"], clean_facts
