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

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from nimbusdesk.agents.specialists import BillingNode, TechnicalNode, escalation_node
from nimbusdesk.agents.state import SupportState
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
from nimbusdesk.rag.ports import Reranker
from nimbusdesk.rag.retrieval import Retriever


def build_support_graph(
    fast_llm: LLMProvider,
    strong_llm: ToolCallingLLM,
    retriever: Retriever,
    reranker: Reranker,
    checkpointer: Any | None = None,
    account_tools: Any | None = None,
) -> CompiledStateGraph:
    """Composition: model tiers follow the same routing logic as everywhere —
    triage runs on the fast tier, customer-facing specialists on the strong.
    `account_tools`: pass MCP-loaded tools to swap the local customer lookup
    for the protocol-backed CRM/ticketing (phase 5)."""
    triage_agent = TriageAgent(fast_llm)

    def triage_node(state: SupportState) -> dict:
        # TriageAgent already degrades every failure to the UNKNOWN fallback,
        # so this node never raises and never blocks the flow.
        return {"triage": triage_agent.triage(state.question)}

    builder = StateGraph(SupportState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("triage", triage_node)
    builder.add_node(
        "technical", TechnicalNode(strong_llm, retriever, reranker, account_tools)
    )
    builder.add_node("billing", BillingNode(strong_llm, retriever, reranker, account_tools))
    builder.add_node("escalation", escalation_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            ROUTE_END: END,
            ROUTE_TRIAGE: "triage",
            ROUTE_TECHNICAL: "technical",
            ROUTE_BILLING: "billing",
            ROUTE_ESCALATION: "escalation",
        },
    )
    # Hub-and-spoke: workers never talk to each other, only to the hub.
    for worker in ("triage", "technical", "billing", "escalation"):
        builder.add_edge(worker, "supervisor")

    return builder.compile(checkpointer=checkpointer)


def run_support_graph(
    graph: CompiledStateGraph,
    question: str,
    customer_email: str | None = None,
    thread_id: str = "default",
) -> SupportState:
    result = graph.invoke(
        SupportState(question=question, customer_email=customer_email),
        config={"configurable": {"thread_id": thread_id}},
    )
    # LangGraph returns plain dict state values; re-validate at the boundary
    # so callers always hold a typed object.
    return SupportState.model_validate(result)
