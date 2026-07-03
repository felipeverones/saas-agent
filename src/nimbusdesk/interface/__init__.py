"""Interface layer — how the outside world talks to NimbusDesk.

TWO ENTRY POINTS, ONE BRAIN
- `api/`: FastAPI app exposing `/chat` (SSE streaming) and ticket endpoints.
  The API is the production-shaped surface: it's what a web frontend, a Zendesk
  plugin or another service would integrate with.
- `cli/`: interactive terminal client (Typer + Rich) that consumes the API.
  It exists for demo/DX — and BECAUSE it goes through the API, demoing the CLI
  also proves the API works. No business logic lives in either entry point.

WHY STREAMING (SSE) FOR /chat
Agent runs take seconds to minutes. Users tolerate latency they can SEE
(tokens streaming, "consulting the knowledge base...") and abandon latency they
can't. Streaming is UX, but in agent products it's load-bearing UX.
"""
