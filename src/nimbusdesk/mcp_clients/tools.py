"""Remote MCP tools as agent capabilities + the consent gate.

This module IS the seam MCP promises: RemoteMcpTool satisfies the same
ToolLike port as a local Tool, so the agent loop cannot tell whether a
capability runs in-process or in another process across the wire.

THE CONSENT GATE (part of the MCP authorization model)
The spec is explicit: the HOST application must obtain user consent before
invoking tools, especially non-read-only ones. We enforce it in the CLIENT
layer — centralized here, no individual agent can forget it:
- read-only tools (server annotation readOnlyHint=true) run freely;
- everything else asks the ConsentPolicy first; a denial becomes a ToolError
  observation, so the MODEL learns the action was refused and can adapt
  (offer an alternative, tell the user) instead of silently failing.
- Missing annotations = consent required (fail closed).
Phase 7 swaps CliConsent for a checkpointed human-in-the-loop interrupt.
"""

import logging
from typing import Protocol

from nimbusdesk.agents.tools import ToolError
from nimbusdesk.llm.ports import ToolSpec
from nimbusdesk.mcp_clients.client import RemoteToolInfo, SyncMcpClient

logger = logging.getLogger(__name__)


class ConsentPolicy(Protocol):
    def approve(self, tool_name: str, arguments: dict) -> bool: ...


class AutoDenyConsent:
    """Safe default: no interactive user available -> sensitive calls refused."""

    def approve(self, tool_name: str, arguments: dict) -> bool:
        return False


class CliConsent:
    """Dev-mode consent: asks on stdin. The UX matters less than the principle:
    the decision happens OUTSIDE the model, in the human's hands."""

    def approve(self, tool_name: str, arguments: dict) -> bool:
        reply = input(f"\n[consent] allow '{tool_name}' with {arguments}? [y/N] ")
        return reply.strip().lower() in ("y", "yes")


class RemoteMcpTool:
    """Adapter: one tool on an MCP server, presented through the ToolLike port.

    Note what is ABSENT: no input_model. Validation happens SERVER-side
    against the same schema we forward to the LLM — the server owns its
    contract, and a validation failure comes back as an error observation
    like any other.
    """

    def __init__(self, client: SyncMcpClient, info: RemoteToolInfo, consent: ConsentPolicy):
        self._client = client
        self._info = info
        self._consent = consent
        self.name = info.name

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._info.name,
            description=self._info.description,
            input_schema=self._info.input_schema,
        )

    def run(self, arguments: dict) -> str:
        if not self._info.read_only:
            if not self._consent.approve(self.name, arguments):
                logger.info("consent denied for %s", self.name)
                raise ToolError(
                    f"the user declined to authorize '{self.name}'; do not retry — "
                    "explain what you wanted to do and ask how to proceed"
                )
        text, is_error = self._client.call_tool(self.name, arguments)
        if is_error:
            raise ToolError(text)
        return text


def load_remote_tools(client: SyncMcpClient, consent: ConsentPolicy) -> list[RemoteMcpTool]:
    """Discovery: ask the server what it offers, adapt everything.

    This one line of dynamism is the M x N payoff — the agent gains whatever
    tools the server ships, today's and future ones, with zero code changes
    on our side.
    """
    return [RemoteMcpTool(client, info, consent) for info in client.list_tools()]
