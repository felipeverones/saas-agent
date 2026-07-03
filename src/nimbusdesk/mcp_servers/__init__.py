"""MCP servers — external systems exposed via the Model Context Protocol.

WHAT MCP IS
An open protocol (introduced by Anthropic in Nov 2024, industry-standard by 2026)
that decouples AI apps from their tools. Think "USB-C for AI": a server exposes
TOOLS (actions), RESOURCES (readable data) and PROMPTS in a standard shape, and
ANY MCP-capable client/agent can use them — no per-app glue code.

WHY IT EXISTS (the M x N problem)
Without a protocol, M apps x N tools = M*N custom integrations. With MCP each
side implements the protocol once: M + N. Same reasoning that gave us HTTP or LSP.

WHAT WE BUILD HERE (phase 5)
Two standalone servers simulating the systems a support platform talks to:
- `crm/`       — mocked customer/subscription data (read-heavy)
- `ticketing/` — CRUD over fake support tickets (read-write)
Each is an independent process speaking Streamable HTTP — the current transport
(the older SSE transport was deprecated in 2025; protocol version is pinned and
commented in each server).

SECURITY NOTES BAKED IN
- Strict Pydantic validation on every tool input: tool results and tool args are
  a documented prompt-injection vector; never trust them implicitly.
- Sensitive tools (e.g. updating customer data) require explicit user consent
  before execution — part of the MCP spec's authorization model (phase 7 wires
  this to human-in-the-loop).
"""
