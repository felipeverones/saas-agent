"""Triage's contract: valid JSON becomes a typed decision; EVERY failure mode
degrades to the same safe fallback (UNKNOWN/0.0) that policy routes to humans."""

from nimbusdesk.agents.triage import FALLBACK, TriageAgent
from nimbusdesk.domain.support import TicketCategory, TicketPriority
from tests.fakes import ExplodingLLMProvider, FakeLLMProvider

GOOD = (
    '{"category": "billing", "priority": "high", '
    '"confidence": 0.9, "summary": "Refund request over limit"}'
)


def test_valid_json_becomes_typed_decision():
    decision = TriageAgent(FakeLLMProvider([GOOD])).triage("I want my money back")
    assert decision.category is TicketCategory.BILLING
    assert decision.priority is TicketPriority.HIGH
    assert decision.confidence == 0.9
    assert decision.is_routable


def test_json_wrapped_in_prose_still_parses():
    decision = TriageAgent(FakeLLMProvider([f"Sure! Here it is:\n{GOOD}\nDone."])).triage("q")
    assert decision.category is TicketCategory.BILLING


def test_unparseable_output_degrades_to_fallback():
    assert TriageAgent(FakeLLMProvider(["it looks like billing?"])).triage("q") == FALLBACK


def test_schema_violation_degrades_to_fallback():
    # Valid JSON, invalid values: category outside the enum, confidence > 1.
    bad = '{"category": "quantum", "priority": "normal", "confidence": 7, "summary": ""}'
    assert TriageAgent(FakeLLMProvider([bad])).triage("q") == FALLBACK


def test_llm_outage_degrades_to_fallback():
    assert TriageAgent(ExplodingLLMProvider()).triage("q") == FALLBACK


def test_fallback_is_never_routable():
    # The safety property everything above relies on.
    assert not FALLBACK.is_routable