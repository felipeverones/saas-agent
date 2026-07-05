"""The support agent's local tools — knowledge search, CRM lookup, status.

"LOCAL" IS THE POINT (and the setup for phase 5)
These tools run in-process with hardcoded fake data. In phase 5 the CRM and
status data move behind MCP servers and the agent reaches them over the
protocol — the Tool interface stays identical, which will demonstrate exactly
what MCP buys: the agent doesn't care where a capability lives.

Note the search tool WRAPS the phase-2 retrieval stack: to an agent, RAG is
just another tool. That's the modern framing — retrieval is a capability the
agent chooses to use, not a pipeline the request is forced through.
"""

import json
from typing import Literal

from pydantic import BaseModel, Field

from nimbusdesk.agents.tools import Tool
from nimbusdesk.rag.ports import Reranker
from nimbusdesk.rag.retrieval import Retriever

# --- knowledge base search -------------------------------------------------


class SearchInput(BaseModel):
    query: str = Field(min_length=2, max_length=300, description="Search query")


class SearchKnowledgeBaseTool(Tool):
    name = "search_knowledge_base"
    description = (
        "Search NimbusDesk's internal documentation (policies, troubleshooting "
        "guides, plan details). Use for ANY question about how the product or "
        "company policies work. Returns the most relevant excerpts with sources."
    )
    input_model = SearchInput

    def __init__(self, retriever: Retriever, reranker: Reranker, top_k: int = 3) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._top_k = top_k

    def execute(self, args: SearchInput) -> str:
        candidates = self._retriever.search(args.query, k=20)
        top = self._reranker.rerank(args.query, candidates, top_n=self._top_k)
        if not top:
            return "no results found"
        return "\n\n".join(
            f"[source: {r.chunk.title} — {r.chunk.section}]\n{r.chunk.text}" for r in top
        )


# --- customer lookup (fake CRM; becomes crm-mcp-server in phase 5) ----------

_FAKE_CUSTOMERS = {
    "dana@acme.io": {
        "name": "Dana Reyes",
        "company": "Acme Corp",
        "plan": "business",
        "seats": 140,
        "status": "active",
        "payment_state": "current",
    },
    "li.wei@bitforge.dev": {
        "name": "Li Wei",
        "company": "BitForge",
        "plan": "pro",
        "seats": 12,
        "status": "active",
        "payment_state": "past_due",
    },
    "sam@nimbusfan.com": {
        "name": "Sam Ortiz",
        "company": "—",
        "plan": "free",
        "seats": 3,
        "status": "active",
        "payment_state": "current",
    },
}


class CustomerLookupInput(BaseModel):
    email: str = Field(
        pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", description="Customer's account email"
    )


class LookupCustomerTool(Tool):
    name = "lookup_customer"
    description = (
        "Look up a customer's account by email: plan, seats, subscription and "
        "payment status. Use before answering anything plan- or billing-specific."
    )
    input_model = CustomerLookupInput

    def execute(self, args: CustomerLookupInput) -> str:
        customer = _FAKE_CUSTOMERS.get(args.email.lower())
        if customer is None:
            # "Not found" is a valid ANSWER, not an error: the agent should
            # relay it, not retry. Compare with ToolError in tools.py.
            return f"no customer found with email {args.email}"
        return json.dumps(customer)


# --- service status (fake; becomes part of phase 5's MCP surface) -----------

_FAKE_STATUS = {
    "web": "operational",
    "api": "operational",
    "sync": "degraded_performance — elevated sync latency in us-east since 09:20 UTC",
    "webhooks": "operational",
    "sso": "operational",
}


class ServiceStatusInput(BaseModel):
    component: Literal["web", "api", "sync", "webhooks", "sso"] = Field(
        description="Which service component to check"
    )


class GetServiceStatusTool(Tool):
    name = "get_service_status"
    description = (
        "Check the current operational status of a NimbusDesk component. Use "
        "when a customer reports something broken or slow, BEFORE deep "
        "troubleshooting — an active incident may already explain the symptom."
    )
    input_model = ServiceStatusInput

    def execute(self, args: ServiceStatusInput) -> str:
        return f"{args.component}: {_FAKE_STATUS[args.component]}"
