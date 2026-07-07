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
| 5 | MCP servers + client | ✅ |
| 6 | Memory: short & long term | ✅ |
| 7 | Guardrails + human-in-the-loop | ✅ |
| 8 | Observability + evaluation suite | ✅ |
| 9 | Packaging: Docker, API + CLI | ✅ |
| 10 | Portfolio polish | ⏳ |

## Run it in 5 minutes

```bash
# prerequisites: Docker + an Anthropic API key
cp .env.example .env          # put your ANTHROPIC_API_KEY in it
docker compose up --build     # qdrant + phoenix + api (auto-ingests the KB)

# then talk to it (needs uv: https://docs.astral.sh/uv/)
uv run nimbus chat --email dana@acme.io
```

You get: triage → supervisor routing → specialist agents with tools and
hybrid RAG, node-by-node SSE progress, refunds over $500 pausing for YOUR
approval in the CLI, memory across sessions, and every request traced at
http://localhost:6006.

## Development quickstart

```bash
make setup   # venv + deps (uv provisions Python 3.12)
make up      # infra only: Qdrant (localhost:6333) + Phoenix (localhost:6006)
make test    # 119 tests — no API keys needed, LLM is always mocked
make run     # API from the venv (hot iteration; then `make cli` to chat)
make ingest  # index the fake NimbusDesk knowledge base into Qdrant
make search Q="customer wants money back after 3 weeks"

# full grounded answer with citations + self-check (needs ANTHROPIC_API_KEY in .env)
make ask Q="can we force two factor authentication for the whole workspace?"

# single support agent: watches service status, looks up customers, searches the KB
make agent Q="dana@acme.io says sync is very slow today, what's going on?"

# full multi-agent team: triage -> supervisor routing -> specialist (or human escalation)
make team Q="I was double charged this month, can I get a refund?"

# MCP: run each server in its own terminal, then point the agents at them (--mcp)
make mcp-crm         # terminal 1
make mcp-ticketing   # terminal 2
uv run python -m nimbusdesk.agents team "what plan is dana@acme.io on?" --mcp

# interactive chat with short-term (thread) and long-term (cross-session) memory
make chat EMAIL=dana@acme.io THREAD=monday
# ... close it, come back "tomorrow" on a NEW thread: the system remembers Dana
make chat EMAIL=dana@acme.io THREAD=tuesday

# observability: set TRACING_ENABLED=true in .env, run anything, then browse
# the span trees at http://localhost:6006 (Arize Phoenix, started by `make up`)

# evaluation: retrieval suite is free; routing/faithfulness run when a key exists
make eval
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
