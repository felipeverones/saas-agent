"""The support graph: supervisor-worker orchestration wired in LangGraph.

TOPOLOGY (hub-and-spoke — every specialist returns to the hub):

                 START
                   |
              [supervisor] --------------------> END
               |   |   |   \\
          [triage] |  [billing]
               |  [technical]  \\
               |   |   |    [escalation]
               +---+---+--------+
               (all edges return to supervisor)

WHY A GRAPH INSTEAD OF NESTED FUNCTION CALLS
1. CHECKPOINTING: LangGraph persists the typed state after EVERY node. A crash
   resumes from the last good step; phase 7's human approval pauses a run for
   days on the same mechanism. A call stack lives in RAM and dies with it.
2. AUDITABILITY: transitions are data — you can replay exactly which nodes ran
   in which order with which state (phase 8 turns this into trace spans).
3. EXPLICIT CONTROL FLOW: the routing table below is the ONLY way agents hand
   off; nobody secretly calls anybody. Compare agents/handoff_demo.py, where
   agents transfer control to each other directly, to feel the difference.

THREADS: a `thread_id` in the config keys the checkpoint history — one thread
per support conversation. Same id later = resume with full state (this is the
short-term memory of phase 6).
"""

import logging
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from nimbusdesk.agents.hitl import human_approval_node
from nimbusdesk.agents.specialists import BillingNode, TechnicalNode, escalation_node
from nimbusdesk.agents.state import ChatTurn, SupportState
from nimbusdesk.agents.supervisor import (
    ROUTE_APPROVAL,
    ROUTE_BILLING,
    ROUTE_END,
    ROUTE_ESCALATION,
    ROUTE_TECHNICAL,
    ROUTE_TRIAGE,
    route_from_supervisor,
    supervisor_node,
)
from nimbusdesk.agents.triage import TriageAgent
from nimbusdesk.guardrails.input_validation import validate_customer_message
from nimbusdesk.llm.ports import LLMProvider, ToolCallingLLM
from nimbusdesk.memory.service import MemoryService
from nimbusdesk.observability.tracing import span
from nimbusdesk.rag.ports import Reranker
from nimbusdesk.rag.retrieval import Retriever

logger = logging.getLogger(__name__)


def build_support_graph(
    fast_llm: LLMProvider,
    strong_llm: ToolCallingLLM,
    retriever: Retriever,
    reranker: Reranker,
    checkpointer: Any | None = None,
    account_tools: Any | None = None,
    memory: MemoryService | None = None,
) -> CompiledStateGraph:
    """Composition: model tiers follow the same routing logic as everywhere —
    triage runs on the fast tier, customer-facing specialists on the strong.
    `account_tools`: pass MCP-loaded tools to swap the local customer lookup
    for the protocol-backed CRM/ticketing (phase 5).
    `memory`: long-term memory service (phase 6); None disables recall/write
    but the graph still keeps short-term history."""
    triage_agent = TriageAgent(fast_llm)

    def triage_node(state: SupportState) -> dict:
        # TriageAgent already degrades every failure to the UNKNOWN fallback,
        # so this node never raises and never blocks the flow.
        return {"triage": triage_agent.triage(state.question)}

    def guard_input_node(state: SupportState) -> dict:
        """The input gate (phase 7). Structural problems reject the turn with
        a polite canned answer; injection-looking content is flagged into the
        state (flag, don't block — see guardrails/input_validation.py)."""
        check = validate_customer_message(state.question)
        if not check.ok:
            return {
                "final_answer": (
                    "Sorry, we couldn't process that message "
                    f"({check.rejection_reason}). Please rephrase and try again."
                ),
                "resolved_by": "input_guard",
            }
        return {"question": check.sanitized, "input_flags": check.flags}

    def route_after_guard(state: SupportState) -> str:
        return "rejected" if state.final_answer is not None else "accepted"

    def recall_node(state: SupportState) -> dict:
        """First node of every turn: load what we know about this customer.
        Fails open — an unavailable memory store degrades the answer's
        personalization, never its availability."""
        if memory is None or not state.customer_email:
            return {}
        try:
            return {"memory_context": memory.recall(state.customer_email, state.question)}
        except Exception as error:
            logger.warning("memory recall failed (%s: %s)", type(error).__name__, error)
            return {}

    def finalize_node(state: SupportState) -> dict:
        """Last node of every turn: append the exchange to short-term history
        (via the reducer) and distill it into long-term memory."""
        if memory is not None and state.customer_email and state.final_answer:
            memory.record_turn(
                email=state.customer_email,
                thread_id=state.thread_hint or "unknown",
                turn_index=state.turn_index,
                question=state.question,
                answer=state.final_answer,
            )
        return {
            "history": [
                ChatTurn(role="customer", content=state.question),
                ChatTurn(role="assistant", content=state.final_answer or ""),
            ],
            "turn_index": state.turn_index + 1,
        }

    def traced(name: str, fn) -> Any:
        """Every graph node gets a span — the top level of the request tree.
        (human_approval is NOT wrapped: interrupt() suspends mid-node, and a
        span held open across a days-long pause would be nonsense telemetry.)
        """

        def wrapper(state: SupportState) -> dict:
            with span(f"graph.{name}"):
                return fn(state)

        return wrapper

    builder = StateGraph(SupportState)
    builder.add_node("guard_input", traced("guard_input", guard_input_node))
    builder.add_node("recall", traced("recall", recall_node))
    builder.add_node("supervisor", traced("supervisor", supervisor_node))
    builder.add_node("triage", traced("triage", triage_node))
    builder.add_node(
        "technical",
        traced("technical", TechnicalNode(strong_llm, retriever, reranker, account_tools)),
    )
    builder.add_node(
        "billing",
        traced("billing", BillingNode(strong_llm, retriever, reranker, account_tools)),
    )
    builder.add_node("escalation", traced("escalation", escalation_node))
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("finalize", traced("finalize", finalize_node))

    builder.add_edge(START, "guard_input")
    builder.add_conditional_edges(
        "guard_input", route_after_guard, {"accepted": "recall", "rejected": "finalize"}
    )
    builder.add_edge("recall", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            ROUTE_END: "finalize",
            ROUTE_TRIAGE: "triage",
            ROUTE_TECHNICAL: "technical",
            ROUTE_BILLING: "billing",
            ROUTE_ESCALATION: "escalation",
            ROUTE_APPROVAL: "human_approval",
        },
    )
    # Hub-and-spoke: workers never talk to each other, only to the hub.
    for worker in ("triage", "technical", "billing", "escalation", "human_approval"):
        builder.add_edge(worker, "supervisor")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


# Given the interrupt payload, returns the human's decision, e.g.
# {"approved": True} or {"approved": False, "note": "outside refund window"}.
ApprovalCallback = Callable[[dict], dict]


def _deny_no_human(payload: dict) -> dict:
    # Fail-closed default: with nobody available to approve, irreversible
    # actions are denied, never silently executed.
    return {"approved": False, "note": "no human reviewer available"}


def run_support_graph(
    graph: CompiledStateGraph,
    question: str,
    customer_email: str | None = None,
    thread_id: str = "default",
    approval_callback: ApprovalCallback = _deny_no_human,
) -> SupportState:
    """Run ONE conversation turn on a thread.

    TURN RESET: with checkpointing, the previous turn's state is restored
    before this input merges in. Per-turn fields (answer, triage, failures,
    budget) must be explicitly reset or the supervisor would see last turn's
    final_answer and end immediately. What we deliberately DON'T reset is the
    short-term memory: `history` (append-only via reducer) and `turn_index`.

    INTERRUPTS: if the run pauses for human approval (large refund), the
    interrupt payload goes to `approval_callback` and the graph resumes with
    its decision. In the CLI that's an interactive prompt; in a real product
    the payload would land in an operator queue and the resume could happen
    days later from another process — the checkpoint doesn't care.
    """
    turn_input: dict = {
        "question": question,
        "thread_hint": thread_id,
        "final_answer": None,
        "resolved_by": None,
        "escalated": False,
        "escalation_reason": None,
        "failures": [],
        "supervisor_visits": 0,
        "triage": None,
        "memory_context": None,
        "input_flags": [],
        "pending_refund": None,
        "refund_decision": None,
    }
    if customer_email is not None:
        turn_input["customer_email"] = customer_email

    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(turn_input, config=config)
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        decision = approval_callback(payload)
        result = graph.invoke(Command(resume=decision), config=config)

    # LangGraph returns plain dict state values; re-validate at the boundary
    # so callers always hold a typed object.
    return SupportState.model_validate(result)
