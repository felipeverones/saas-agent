"""Domain layer — pure business rules. The only layer with ZERO AI in it.

WHAT LIVES HERE
Pydantic models and plain functions describing the *business* of customer support:
`Ticket`, `Customer`, `TriageDecision`, `Citation`, escalation policies, priority
rules. Things that would still be true if we replaced every LLM with a human.

THE ONE RULE (enforced by tests/unit/test_architecture.py)
This package may import Pydantic and the standard library. It may NOT import
langgraph, anthropic, qdrant, fastapi, mcp — nor any other nimbusdesk package.

WHY SO STRICT
1. Testability: business rules ("a refund over $500 always needs human approval")
   are tested in microseconds with no mocks, no API keys, no network.
2. Stability: frameworks churn yearly; the meaning of a "ticket" doesn't. The most
   stable code must not depend on the most volatile code.
3. Interview answer: when asked "how do you keep an LLM app maintainable?", the
   answer is exactly this — the LLM is an implementation detail at the edge of the
   system, not the center of it.
"""
