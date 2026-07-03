# LEARNING.md — concept review notes

One section per topic: **what it is → where it lives in this repo → why it matters
in production.** Filled in at the end of each phase; review this file before
interviews. (Sections marked ⏳ are placeholders for upcoming phases.)

---

## Clean / layered architecture for LLM apps (Phase 0)

**What it is.** The code is organized in circles: `domain/` (pure business rules)
at the center, `infrastructure/` (vendors: Anthropic, Qdrant, SQLite) at the edge,
and the dependency rule that imports only point inward. Inner layers define the
interfaces ("ports") they need; the edge provides implementations ("adapters").

**Where.** Package docstrings in every `src/nimbusdesk/*/__init__.py`;
the dependency rule is enforced by `tests/unit/test_architecture.py`;
decisions recorded in `docs/ARCHITECTURE.md` (ADR-13 especially).

**Why it matters in production.** The LLM stack churns yearly (frameworks,
providers, protocols). Teams that welded business logic to a framework rewrote
their product twice between 2023 and 2026; teams that isolated it swapped one
adapter. It's also what makes the test suite free: the whole app runs against a
fake LLM provider, so CI needs no API keys and costs zero tokens.

**Talking point.** "My domain layer has an automated test proving it imports no
AI framework — architecture enforced by CI, not by code review vigilance."

## Architecture Decision Records (Phase 0)

**What it is.** A short log entry per non-obvious choice: context, decision,
rejected alternative, consequences.

**Where.** `docs/ARCHITECTURE.md`, ADRs 01-13.

**Why it matters in production.** Six months later nobody remembers *why* Qdrant
and not Chroma. ADRs make trade-off reasoning — the #1 senior-interview signal —
a written artifact you can walk through.

---

## ⏳ RAG — ingestion, chunking, embeddings, retrieval (Phase 1)
## ⏳ Agentic RAG — hybrid search, reranking, query rewriting, self-check (Phase 2)
## ⏳ The agent loop — reason → act → observe (Phase 3)
## ⏳ Multi-agent orchestration — supervisor-worker, handoffs, shared state (Phase 4)
## ⏳ MCP — servers, clients, consent model (Phase 5)
## ⏳ Memory — short-term vs long-term (Phase 6)
## ⏳ Guardrails — I/O validation, HITL, prompt-injection defense (Phase 7)
## ⏳ Observability & evaluation — traces, golden datasets, cost (Phase 8)
