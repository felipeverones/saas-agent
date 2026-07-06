"""MCP integration: real servers + real client, connected in-process.

`create_connected_server_and_client_session` (the SDK's own testing utility)
wires our FastMCP servers to a real ClientSession over memory streams — the
FULL protocol runs (initialize, list_tools, call_tool, server-side schema
validation), just with no sockets. The only untested surface left is the HTTP
transport itself, which is the SDK's code, not ours.
"""

from contextlib import asynccontextmanager

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from nimbusdesk.agents.react import ReactAgent
from nimbusdesk.agents.tools import ToolError
from nimbusdesk.llm.ports import AssistantTurn, ToolCall
from nimbusdesk.mcp_clients.client import SyncMcpClient
from nimbusdesk.mcp_clients.tools import AutoDenyConsent, RemoteMcpTool, load_remote_tools
from nimbusdesk.mcp_servers.crm.server import mcp as crm_server
from nimbusdesk.mcp_servers.ticketing.server import mcp as ticketing_server
from tests.fakes import FakeToolLLM


class InProcessMcpClient(SyncMcpClient):
    """Same client logic, memory transport instead of HTTP (test seam)."""

    def __init__(self, fastmcp_server) -> None:
        super().__init__(url="memory://in-process")
        self._server = fastmcp_server._mcp_server  # the underlying low-level server

    @asynccontextmanager
    async def _connect(self):
        async with create_connected_server_and_client_session(self._server) as session:
            yield session


class ApproveAll:
    def approve(self, tool_name: str, arguments: dict) -> bool:
        return True


@pytest.fixture()
def crm() -> InProcessMcpClient:
    return InProcessMcpClient(crm_server)


@pytest.fixture()
def ticketing() -> InProcessMcpClient:
    return InProcessMcpClient(ticketing_server)


def test_crm_advertises_tools_with_consent_annotations(crm):
    tools = {t.name: t for t in crm.list_tools()}
    assert tools["lookup_customer"].read_only is True
    assert tools["update_customer_plan"].read_only is False
    assert "email" in tools["lookup_customer"].input_schema["properties"]


def test_lookup_customer_found_and_not_found(crm):
    text, is_error = crm.call_tool("lookup_customer", {"email": "dana@acme.io"})
    assert not is_error and "Acme Corp" in text
    text, _ = crm.call_tool("lookup_customer", {"email": "ghost@nowhere.io"})
    assert "no customer found" in text


def test_server_side_schema_validation_rejects_bad_input(crm):
    # The malformed email never reaches the tool function — FastMCP validates
    # against the generated schema first. This is the injection baseline.
    text, is_error = crm.call_tool("lookup_customer", {"email": "not-an-email"})
    assert is_error


def test_sensitive_tool_is_blocked_without_consent(crm):
    [update_tool] = [
        RemoteMcpTool(crm, info, AutoDenyConsent())
        for info in crm.list_tools()
        if info.name == "update_customer_plan"
    ]
    with pytest.raises(ToolError, match="declined"):
        update_tool.run({"email": "dana@acme.io", "new_plan": "enterprise"})


def test_sensitive_tool_runs_with_consent(crm):
    [update_tool] = [
        RemoteMcpTool(crm, info, ApproveAll())
        for info in crm.list_tools()
        if info.name == "update_customer_plan"
    ]
    result = update_tool.run({"email": "sam@nimbusfan.com", "new_plan": "pro"})
    assert '"new_plan": "pro"' in result


def test_ticket_create_and_fetch_roundtrip(ticketing):
    created, is_error = ticketing.call_tool(
        "create_ticket",
        {"email": "dana@acme.io", "subject": "API 429 storms", "body": "since noon"},
    )
    assert not is_error and "TKT-" in created

    ticket_id = created.split('"id": "')[1].split('"')[0]
    fetched, _ = ticketing.call_tool("get_ticket", {"ticket_id": ticket_id})
    assert "API 429 storms" in fetched


def test_agent_uses_remote_mcp_tool_transparently(crm):
    """The finale: the phase-3 agent loop, unchanged, driving a tool that
    lives in another 'system' over the MCP protocol."""
    tools = load_remote_tools(crm, AutoDenyConsent())
    llm = FakeToolLLM(
        [
            AssistantTurn(
                tool_calls=[
                    ToolCall(id="c1", name="lookup_customer", arguments={"email": "dana@acme.io"})
                ]
            ),
            AssistantTurn(text="Dana is on the Business plan."),
        ]
    )
    result = ReactAgent(llm, tools, "support agent").run("what plan is dana@acme.io on?")

    assert result.answer == "Dana is on the Business plan."
    assert not result.steps[0].is_error
    assert "Acme Corp" in result.steps[0].observation