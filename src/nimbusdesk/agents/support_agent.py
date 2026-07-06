"""The NimbusDesk support agent: system prompt + assembly.

PROMPT DESIGN NOTES (the parts that earn their keep)
- Role + boundaries first: models follow persona instructions more reliably
  than scattered rules.
- Tool USAGE POLICY in the prompt, not just tool descriptions: descriptions
  say what a tool does; the system prompt says how this agent should combine
  them (status before troubleshooting, KB before answering policy questions).
- Honesty rules mirror the RAG pipeline: no invented policies, cite what the
  KB search returned, admit when information is missing.
"""

from typing import Sequence

from nimbusdesk.agents.local_tools import (
    GetServiceStatusTool,
    LookupCustomerTool,
    SearchKnowledgeBaseTool,
)
from nimbusdesk.agents.react import ReactAgent
from nimbusdesk.agents.tools import ToolLike
from nimbusdesk.llm.ports import ToolCallingLLM
from nimbusdesk.rag.ports import Reranker
from nimbusdesk.rag.retrieval import Retriever

SUPPORT_SYSTEM_PROMPT = """You are a support agent for NimbusDesk, a cloud \
workspace product.

How to work:
- For anything about product behavior, plans or policies: search the knowledge \
base FIRST and base your answer on what it returns, mentioning the source \
document. Never invent policy details.
- When a customer reports something broken or slow: check the relevant \
component's service status BEFORE troubleshooting — an active incident may \
already explain it.
- When the question involves a specific customer's account or billing: look \
the customer up by email first. If no email was provided, ask for it instead \
of guessing.
- If the knowledge base and tools don't give you enough to answer safely, say \
so and offer to escalate to a human agent.
- Be concise, factual and friendly. Plain text only."""


def build_support_agent(
    llm: ToolCallingLLM,
    retriever: Retriever,
    reranker: Reranker,
    max_iterations: int = 8,
    account_tools: Sequence[ToolLike] | None = None,
) -> ReactAgent:
    """Composition helper: same wiring for the CLI and for tests.

    `account_tools` is the phase-5 seam: by default the customer lookup is
    the local in-process tool; pass MCP-loaded tools instead and the agent
    reaches the CRM/ticketing systems over the protocol — with zero changes
    to the agent itself. That swap being trivial IS the demo.
    """
    account = list(account_tools) if account_tools is not None else [LookupCustomerTool()]
    return ReactAgent(
        llm=llm,
        tools=[
            SearchKnowledgeBaseTool(retriever, reranker),
            *account,
            GetServiceStatusTool(),
        ],
        system_prompt=SUPPORT_SYSTEM_PROMPT,
        max_iterations=max_iterations,
    )
