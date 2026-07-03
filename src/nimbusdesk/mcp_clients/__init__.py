"""MCP client layer — how our agents CONSUME the MCP servers.

WHY A DEDICATED PACKAGE
The whole value of MCP is the seam between "agent" and "tool provider". This
package IS that seam: it connects to servers, lists their tools, adapts them into
the tool interface our agents use, and enforces the consent gate for sensitive
calls. Agents never import server code — if they did, we'd be back to hardwired
integrations and MCP would be decorative.

Includes the CONSENT WRAPPER: tools flagged as sensitive (mutations, PII access)
are intercepted here and require explicit approval before the call crosses the
wire. Centralizing it in the client means no individual agent can forget it.
"""
