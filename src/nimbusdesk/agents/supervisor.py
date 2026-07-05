"""Supervisor — the routing policy at the hub of the agent graph.

DESIGN CHOICE WORTH DEFENDING: A CODE POLICY, NOT AN LLM ROUTER
Two ways to build a supervisor:
1. LLM-router: a model reads the state and picks the next agent. Flexible,
   handles cases nobody anticipated — but unauditable ("why did it route
   there?"), non-deterministic, and costs a call per hop.
2. Policy-router (ours): the LLM's judgment is captured ONCE as structured
   data (TriageDecision, with confidence), and routing over that data is
   plain code — deterministic, unit-testable branch by branch, free.

We push the intelligence to the EDGE (triage) and keep the CENTER dumb and
auditable. For a support desk with four known lanes, that's the right
trade-off; an open-ended assistant with dozens of dynamic skills would justify
the LLM-router. Knowing which regime you're in is the senior answer.

The routing rules, in priority order:
    answer ready                 -> end
    budget exhausted / failure   -> escalation
    no triage yet                -> triage
    untrustworthy triage         -> escalation (never guess on low confidence)
    urgent priority              -> escalation (business rule: humans own P1s)
    category                     -> technical | billing
"""

from nimbusdesk.agents.state import SupportState
from nimbusdesk.domain.support import TicketCategory, TicketPriority

# Hub-visit budget: every specialist hop returns here, so this bounds the
# whole graph — same principle as ReactAgent.max_iterations one level down.
MAX_SUPERVISOR_VISITS = 6

# Node names double as routing targets; END is LangGraph's terminal sentinel.
ROUTE_END = "end"
ROUTE_TRIAGE = "triage"
ROUTE_TECHNICAL = "technical"
ROUTE_BILLING = "billing"
ROUTE_ESCALATION = "escalation"


def supervisor_node(state: SupportState) -> dict:
    """The node only does bookkeeping; routing happens in the conditional
    edge (route_from_supervisor) over the updated state."""
    return {"supervisor_visits": state.supervisor_visits + 1}


def route_from_supervisor(state: SupportState) -> str:
    if state.final_answer is not None:
        return ROUTE_END

    if state.escalated:
        # Escalation already produced its handoff message — nothing else to do.
        return ROUTE_END

    if state.supervisor_visits > MAX_SUPERVISOR_VISITS:
        return ROUTE_ESCALATION

    if state.failures:
        # A specialist crashed. Policy: don't retry blindly (the crash will
        # usually repeat), hand to a human with the failure recorded.
        return ROUTE_ESCALATION

    if state.triage is None:
        return ROUTE_TRIAGE

    if not state.triage.is_routable:
        return ROUTE_ESCALATION

    if state.triage.priority is TicketPriority.URGENT:
        return ROUTE_ESCALATION

    if state.triage.category is TicketCategory.BILLING:
        return ROUTE_BILLING

    # TECHNICAL and ACCOUNT both go to the technical specialist: account
    # questions (SSO, permissions) are covered by the same KB + tools.
    return ROUTE_TECHNICAL
