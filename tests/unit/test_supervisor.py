"""The routing policy, branch by branch — the payoff of a CODE supervisor over
an LLM router: every routing decision is a deterministic, testable fact."""

from nimbusdesk.agents.state import SupportState
from nimbusdesk.agents.supervisor import (
    MAX_SUPERVISOR_VISITS,
    ROUTE_BILLING,
    ROUTE_END,
    ROUTE_ESCALATION,
    ROUTE_TECHNICAL,
    ROUTE_TRIAGE,
    route_from_supervisor,
)
from nimbusdesk.domain.support import TicketCategory, TicketPriority, TriageDecision


def _state(**kwargs) -> SupportState:
    kwargs.setdefault("supervisor_visits", 1)
    return SupportState(question="q", **kwargs)


def _triage(category=TicketCategory.TECHNICAL, priority=TicketPriority.NORMAL, confidence=0.9):
    return TriageDecision(category=category, priority=priority, confidence=confidence)


def test_answer_ready_ends():
    assert route_from_supervisor(_state(final_answer="done")) == ROUTE_END


def test_no_triage_yet_goes_to_triage():
    assert route_from_supervisor(_state()) == ROUTE_TRIAGE


def test_budget_exhaustion_escalates():
    state = _state(supervisor_visits=MAX_SUPERVISOR_VISITS + 1)
    assert route_from_supervisor(state) == ROUTE_ESCALATION


def test_specialist_failure_escalates():
    state = _state(failures=["technical: ConnectionError: boom"], triage=_triage())
    assert route_from_supervisor(state) == ROUTE_ESCALATION


def test_low_confidence_triage_escalates():
    state = _state(triage=_triage(confidence=0.3))
    assert route_from_supervisor(state) == ROUTE_ESCALATION


def test_unknown_category_escalates():
    state = _state(triage=_triage(category=TicketCategory.UNKNOWN, confidence=0.9))
    assert route_from_supervisor(state) == ROUTE_ESCALATION


def test_urgent_priority_escalates_even_when_confident():
    state = _state(triage=_triage(priority=TicketPriority.URGENT))
    assert route_from_supervisor(state) == ROUTE_ESCALATION


def test_billing_routes_to_billing():
    state = _state(triage=_triage(category=TicketCategory.BILLING))
    assert route_from_supervisor(state) == ROUTE_BILLING


def test_technical_and_account_route_to_technical():
    for category in (TicketCategory.TECHNICAL, TicketCategory.ACCOUNT):
        assert route_from_supervisor(_state(triage=_triage(category=category))) == (
            ROUTE_TECHNICAL
        )