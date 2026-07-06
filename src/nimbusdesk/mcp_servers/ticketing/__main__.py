"""Run the ticketing MCP server: `python -m nimbusdesk.mcp_servers.ticketing`.

Streamable HTTP — listens at http://127.0.0.1:8102/mcp.
"""

from nimbusdesk.mcp_servers.ticketing.server import mcp

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
