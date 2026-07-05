"""Specialist nodes: technical, billing, escalation.

COMPOSITION NOTE: the specialists REUSE the phase-3 ReactAgent — a node in the
graph can itself be a full agent with its own inner tool loop. That nesting
(graph orchestrates agents; each agent runs its own ReAct loop) is the
standard shape of production multi-agent systems: LangGraph governs the
BETWEEN-agents flow, ReAct governs the WITHIN-agent flow.

FAILURE CONTRACT: nodes never raise. Any exception is caught and recorded in
`state.failures`; the supervisor's policy then routes to escalation. One
crashed specialist degrades the ticket to a human handoff — it must never
kill the whole run (requirement: agent failure without derrubar o fluxo).
"""

import logging

from nimbusdesk.agents.local_tools import LookupCustomerTool, SearchKnowledgeBaseTool
from nimbusdesk.agents.react import ReactAgent
from nimbusdesk.agents.state import SupportState
from nimbusdesk.agents.support_agent import build_support_agent
from nimbusdesk.llm.ports import ToolCallingLLM
from nimbusdesk.rag.ports import Reranker
from nimbusdesk.rag.retrieval import Retriever

logger = logging.getLogger(__name__)

BILLING_SYSTEM_PROMPT = """You are a billing specialist for NimbusDesk.

How to work:
- For policy questions (refund windows, proration, dunning): search the \
knowledge base and answer from what it returns, citing the source document.
- For account-specific questions: look the customer up by email first. If no \
email is available, ask for it.
- You can EXPLAIN charges and policies, but you cannot issue refunds or \
change subscriptions — for those, tell the customer a specialist will follow \
up. Never promise a refund.
- Be concise, factual and friendly. Plain text only."""


def _question_with_context(state: SupportState) -> str:
    parts = [state.question]
    if state.customer_email:
        parts.append(f"(customer email on file: {state.customer_email})")
    if state.triage:
        parts.append(f"(triage summary: {state.triage.summary})")
    return "\n".join(parts)


class TechnicalNode:
    """Wraps the phase-3 support agent (KB search + status + customer lookup)."""

    def __init__(self, llm: ToolCallingLLM, retriever: Retriever, reranker: Reranker) -> None:
        self._agent = build_support_agent(llm, retriever, reranker)

    def __call__(self, state: SupportState) -> dict:
        try:
            result = self._agent.run(_question_with_context(state))
        except Exception as error:
            logger.exception("technical specialist failed")
            return {"failures": [*state.failures, f"technical: {type(error).__name__}: {error}"]}
        if result.hit_iteration_limit:
            return {"failures": [*state.failures, "technical: iteration limit reached"]}
        return {"final_answer": result.answer, "resolved_by": "technical"}


class BillingNode:
    def __init__(self, llm: ToolCallingLLM, retriever: Retriever, reranker: Reranker) -> None:
        self._agent = ReactAgent(
            llm=llm,
            tools=[SearchKnowledgeBaseTool(retriever, reranker), LookupCustomerTool()],
            system_prompt=BILLING_SYSTEM_PROMPT,
        )

    def __call__(self, state: SupportState) -> dict:
        try:
            result = self._agent.run(_question_with_context(state))
        except Exception as error:
            logger.exception("billing specialist failed")
            return {"failures": [*state.failures, f"billing: {type(error).__name__}: {error}"]}
        if result.hit_iteration_limit:
            return {"failures": [*state.failures, "billing: iteration limit reached"]}
        return {"final_answer": result.answer, "resolved_by": "billing"}


def escalation_node(state: SupportState) -> dict:
    """Deterministic by design: when the system is already unsure (or broken),
    the LAST thing you want is another LLM call that can also fail. It formats
    a handoff summary for the human queue. Phase 7 upgrades this node with
    interrupt() so a human approves/answers INSIDE the paused graph."""
    reasons = []
    if state.failures:
        reasons.append(f"specialist failure(s): {'; '.join(state.failures)}")
    if state.triage and not state.triage.is_routable:
        reasons.append(
            f"low-confidence triage ({state.triage.confidence:.2f}, "
            f"category={state.triage.category})"
        )
    if state.triage and state.triage.priority.value == "urgent":
        reasons.append("urgent priority — human ownership required")
    if state.supervisor_visits > 1 and not reasons:
        reasons.append("routing budget exhausted")
    reason = "; ".join(reasons) or "unspecified"

    summary = state.triage.summary if state.triage else state.question[:200]
    answer = (
        "This ticket has been escalated to a human specialist.\n"
        f"Reason: {reason}\n"
        f"Ticket summary: {summary}\n"
        "The customer will receive a personal follow-up shortly."
    )
    return {
        "final_answer": answer,
        "resolved_by": "escalation",
        "escalated": True,
        "escalation_reason": reason,
    }
