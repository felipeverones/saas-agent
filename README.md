# NimbusDesk 🌩️

A **production-style AI support platform** for a fictional SaaS company — built as
a hands-on learning project covering what the market expects from AI engineers in
2026: multi-agent orchestration (LangGraph), agentic RAG, **MCP** servers/clients,
short/long-term memory, guardrails with human-in-the-loop, and OpenTelemetry
observability.

> 🚧 **Work in progress** — built in phases. Current status below.

| Phase | Scope | Status |
|---|---|---|
| 0 | Architecture, ADRs, scaffold | ✅ |
| 1 | RAG: ingestion + vector retrieval | ⏳ |
| 2 | Agentic RAG: hybrid search, rerank, self-check, citations | ⏳ |
| 3 | Single agent with tools (ReAct loop) | ⏳ |
| 4 | Multi-agent: supervisor + specialists | ⏳ |
| 5 | MCP servers + client | ⏳ |
| 6 | Memory: short & long term | ⏳ |
| 7 | Guardrails + human-in-the-loop | ⏳ |
| 8 | Observability + evaluation suite | ⏳ |
| 9 | Packaging: Docker, API + CLI | ⏳ |
| 10 | Portfolio polish | ⏳ |

## Quickstart (current state)

```bash
# prerequisites: uv (https://docs.astral.sh/uv/), Docker
make setup   # venv + deps (uv provisions Python 3.12)
make up      # Qdrant (localhost:6333) + Phoenix traces (localhost:6006)
make test    # test suite — no API keys needed, LLM is always mocked
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — diagrams + all decisions as ADRs
  (each with the rejected alternative)
- [docs/LEARNING.md](docs/LEARNING.md) — concept-by-concept study notes
- [docs/INTERVIEW_NOTES.md](docs/INTERVIEW_NOTES.md) — Q&A prep with trade-offs

Every package in `src/nimbusdesk/` opens with a docstring explaining the concept
it implements and why it exists — the codebase doubles as course material.
