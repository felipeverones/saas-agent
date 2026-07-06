"""Run the CRM MCP server: `python -m nimbusdesk.mcp_servers.crm`.

Streamable HTTP transport — the server listens at http://127.0.0.1:8101/mcp
and any MCP-capable client (ours, Claude Desktop, an IDE) can connect.
"""

from nimbusdesk.mcp_servers.crm.server import mcp

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
