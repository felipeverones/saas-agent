"""crm-mcp-server — mocked CRM (customers, plans, subscription status) over MCP.

Runs as its own process: `python -m nimbusdesk.mcp_servers.crm`.
Kept deliberately independent from the rest of nimbusdesk (it simulates a system
another team owns — importing our agents from here would defeat the point of MCP).
"""
