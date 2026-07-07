"""MemoryService — the single facade the agent graph talks to.

The graph shouldn't know that long-term memory is two stores and an extractor;
it knows two verbs: `recall(email, question)` before working a ticket, and
`record_turn(...)` after finishing one. Keeping the surface this small is what
will let us swap the internals (e.g. add memory decay, or replace the home-
grown pipeline with Mem0) without touching orchestration.
"""

from nimbusdesk.memory.episodic import EpisodicMemoryStore
from nimbusdesk.memory.profile_store import SqliteProfileStore
from nimbusdesk.memory.writer import MemoryWriter


class MemoryService:
    def __init__(
        self,
        profile_store: SqliteProfileStore,
        episodic_store: EpisodicMemoryStore,
        writer: MemoryWriter,
    ) -> None:
        self._profiles = profile_store
        self._episodes = episodic_store
        self._writer = writer

    def recall(self, email: str, question: str) -> str | None:
        """Everything we know that's relevant, formatted for a system prompt.
        Returns None when we know nothing — callers skip the section entirely
        (an empty 'known facts' header just invites hallucination)."""
        profile = self._profiles.get_profile(email)
        episodes = self._episodes.recall(email, question)

        if not profile and not episodes:
            return None

        lines = [f"What we know about {email} from previous interactions:"]
        if profile:
            lines.append("Profile facts:")
            lines.extend(f"- {key}: {value}" for key, value in profile.items())
        if episodes:
            lines.append("Relevant past interactions:")
            lines.extend(f"- {e.summary}" for e in episodes)
        return "\n".join(lines)

    def record_turn(
        self, email: str, thread_id: str, turn_index: int, question: str, answer: str
    ) -> None:
        self._writer.record_turn(email, thread_id, turn_index, question, answer)
