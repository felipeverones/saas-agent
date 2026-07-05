"""Domain models for support ticket triage and routing.

These are BUSINESS concepts: a human triage team categorized tickets by type,
priority and their own confidence long before LLMs existed. Which is exactly
why they live in domain/ — the rules below (e.g. "low-confidence triage must
be escalated") hold no matter what technology does the triaging.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class TicketCategory(StrEnum):
    TECHNICAL = "technical"
    BILLING = "billing"
    ACCOUNT = "account"
    UNKNOWN = "unknown"


class TicketPriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


# Below this confidence, routing a ticket automatically is riskier than the
# cost of a human look. A BUSINESS threshold, not an ML detail — which is why
# it's defined here and the supervisor merely enforces it.
MIN_ROUTING_CONFIDENCE = 0.5


class TriageDecision(BaseModel):
    """The structured verdict of ticket triage.

    `confidence` is the triager's own estimate that the categorization is
    right. Structured output with an explicit confidence is what lets the
    system know when NOT to trust itself — free-text triage can't do that.
    """

    category: TicketCategory
    priority: TicketPriority = TicketPriority.NORMAL
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = ""

    @property
    def is_routable(self) -> bool:
        return (
            self.category is not TicketCategory.UNKNOWN
            and self.confidence >= MIN_ROUTING_CONFIDENCE
        )
