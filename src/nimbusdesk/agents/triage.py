"""Triage agent — classifies a ticket into a STRUCTURED TriageDecision.

STRUCTURED OUTPUT AS A GUARDRAIL (preview of phase 7)
This agent's output triggers a routing decision, so free text is not
acceptable: the LLM must produce JSON that validates against TriageDecision.
The failure policy is the interesting part — ANY problem (LLM outage,
unparseable reply, invalid values) degrades to the same safe value:

    TriageDecision(category=UNKNOWN, confidence=0.0)

which the supervisor's policy routes to human escalation. Bad classification
never crashes the flow AND never gets silently trusted; it becomes a routable
fact. Cheap model tier: classification is mechanical (see llm/__init__.py).
"""

import logging

from pydantic import ValidationError

from nimbusdesk.domain.support import TicketCategory, TriageDecision
from nimbusdesk.llm.json_parsing import extract_json_object
from nimbusdesk.llm.ports import LLMProvider, Message

logger = logging.getLogger(__name__)

_SYSTEM = """You triage support tickets for NimbusDesk, a cloud workspace \
product.

Classify the ticket and reply with ONLY this JSON:
{"category": "technical" | "billing" | "account", \
"priority": "low" | "normal" | "high" | "urgent", \
"confidence": 0.0-1.0, "summary": "<one sentence>"}

Guidance:
- technical: sync, API, webhooks, performance, errors, integrations
- billing: charges, refunds, invoices, plan changes, payment failures
- account: login, SSO, permissions, member management, workspace settings
- priority urgent ONLY for suspected data loss, security incidents, or a
  fully blocked workspace
- confidence reflects how sure you are of the CATEGORY; use below 0.5 when
  the ticket is ambiguous or spans multiple areas."""

FALLBACK = TriageDecision(category=TicketCategory.UNKNOWN, confidence=0.0)


class TriageAgent:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def triage(self, question: str) -> TriageDecision:
        try:
            completion = self._llm.complete(
                system=_SYSTEM,
                messages=[Message(role="user", content=question)],
                max_tokens=200,
            )
        except Exception as error:
            logger.warning("triage LLM call failed (%s: %s)", type(error).__name__, error)
            return FALLBACK

        data = extract_json_object(completion.text)
        if data is None:
            logger.warning("triage returned unparseable output: %.120s", completion.text)
            return FALLBACK
        try:
            return TriageDecision.model_validate(data)
        except ValidationError as error:
            logger.warning("triage JSON failed schema validation: %s", error)
            return FALLBACK
