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
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from nimbusdesk.agents.specialists import BillingNode, TechnicalNode, escalation_node
from nimbusdesk.agents.state import ChatTurn, SupportState
from nimbusdesk.agents.supervisor import (
    ROUTE_BILLING,
    ROUTE_END,
    ROUTE_ESCALATION,
    ROUTE_TECHNICAL,
    ROUTE_TRIAGE,
    route_from_supervisor,
    supervisor_node,
)
from nimbusdesk.agents.triage import TriageAgent
from nimbusdesk.llm.ports import LLMProvider, ToolCallingLLM
from nimbusdesk.memory.service import MemoryService
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

    builder = StateGraph(SupportState)
    builder.add_node("recall", recall_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("triage", triage_node)
    builder.add_node(
        "technical", TechnicalNode(strong_llm, retriever, reranker, account_tools)
    )
    builder.add_node("billing", BillingNode(strong_llm, retriever, reranker, account_tools))
    builder.add_node("escalation", escalation_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "recall")
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
        },
    )
    # Hub-and-spoke: workers never talk to each other, only to the hub.
    for worker in ("triage", "technical", "billing", "escalation"):
        builder.add_edge(worker, "supervisor")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


def run_support_graph(
    graph: CompiledStateGraph,
    question: str,
    customer_email: str | None = None,
    thread_id: str = "default",
) -> SupportState:
    """Run ONE conversation turn on a thread.

    TURN RESET: with checkpointing, the previous turn's state is restored
    before this input merges in. Per-turn fields (answer, triage, failures,
    budget) must be explicitly reset or the supervisor would see last turn's
    final_answer and end immediately. What we deliberately DON'T reset is the
    short-term memory: `history` (append-only via reducer) and `turn_index`.
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
    }
    if customer_email is not None:
        turn_input["customer_email"] = customer_email

    result = graph.invoke(turn_input, config={"configurable": {"thread_id": thread_id}})
    # LangGraph returns plain dict state values; re-validate at the boundary
    # so callers always hold a typed object.
    return SupportState.model_validate(result)
