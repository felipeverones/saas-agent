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
| 1 | RAG: ingestion + vector retrieval | ✅ |
| 2 | Agentic RAG: hybrid search, rerank, self-check, citations | ✅ |
| 3 | Single agent with tools (ReAct loop) | ✅ |
| 4 | Multi-agent: supervisor + specialists | ✅ |
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
make ingest  # index the fake NimbusDesk knowledge base into Qdrant
make search Q="customer wants money back after 3 weeks"

# full grounded answer with citations + self-check (needs ANTHROPIC_API_KEY in .env)
make ask Q="can we force two factor authentication for the whole workspace?"

# single support agent: watches service status, looks up customers, searches the KB
make agent Q="dana@acme.io says sync is very slow today, what's going on?"

# full multi-agent team: triage -> supervisor routing -> specialist (or human escalation)
make team Q="I was double charged this month, can I get a refund?"
```

Unit tests use fakes (instant, offline); integration tests use real embeddings
plus an in-process Qdrant — run only those with `pytest -m integration`.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — diagrams + all decisions as ADRs
  (each with the rejected alternative)
- [docs/GLOSSARY.md](docs/GLOSSARY.md) — every concept (BM25, RRF, ports,
  cross-encoders…) in plain words, with pointers into the code
- [docs/LEARNING.md](docs/LEARNING.md) — concept-by-concept study notes
- [docs/INTERVIEW_NOTES.md](docs/INTERVIEW_NOTES.md) — Q&A prep with trade-offs

Every package in `src/nimbusdesk/` opens with a docstring explaining the concept
it implements and why it exists — the codebase doubles as course material.
