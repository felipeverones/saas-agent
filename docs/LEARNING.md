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

## RAG part 1 — ingestion, chunking, embeddings, retrieval (Phase 1)

**What it is.** RAG = answer from retrieved evidence instead of model memory.
The ingestion pipeline (`load -> chunk -> embed -> index`) converts documents
into searchable vectors once; at question time, the query is embedded and the
nearest chunks are returned.

**The vocabulary, quickly.**
- *Embedding*: a model maps text to a vector (here: 384 numbers) where similar
  meanings land close together. "money back" and "refund" end up neighbors
  even with zero shared words — that's why this beats keyword search alone.
- *Chunk*: the retrieval unit. We split on markdown headings first (authors
  already segmented by topic), size-split only oversized sections, and keep a
  paragraph of overlap so no sentence loses its context at a cut point.
- *Asymmetric retrieval*: queries and passages are embedded through different
  methods (`embed_query` vs `embed_passages`) because the model was trained to
  place short questions near the long passages that answer them. Using one
  `embed()` for both is a classic silent-quality bug.
- *Contextual enrichment (lite)*: we embed "title — section \n text" but store
  raw text — a cut chunk regains its topic signal, citations stay verbatim.

**Where.** Domain contracts: `src/nimbusdesk/domain/knowledge.py`. Pipeline:
`src/nimbusdesk/rag/` (loading, chunking, ingestion, retrieval, ports).
Adapters: `src/nimbusdesk/infrastructure/{embeddings,vector_store}.py`.
Quality tests: `tests/integration/test_rag_pipeline.py`.

**Why it matters in production.**
- Deterministic chunk ids (UUID5) + upsert make re-ingestion idempotent — you
  can re-run the indexer on every deploy without duplicating the corpus.
- The two-tier test strategy mirrors production reality: fakes prove the
  MECHANICS in milliseconds; a small integration suite with the real model
  proves retrieval QUALITY ("does the user's phrasing find the right doc?").
- Vector params (cosine, named vectors) live in ONE adapter; phase 2 adds
  sparse vectors to the same collection with zero migration.

**Talking point.** "My chunker is structure-aware, my chunk ids are
deterministic for idempotent re-ingestion, and retrieval quality is pinned by
integration tests that ask questions in user language, not doc language."

## RAG part 2 — agentic RAG: hybrid, rerank, rewrite, self-check (Phase 2)

**What it is.** Naive RAG is a fixed pipe (embed -> top-k -> answer). Agentic
RAG adds control points where the system inspects and corrects itself:

- *Query rewriting* (`rag/rewrite.py`): a cheap LLM turns "hey can we force
  everyone onto 2fa?" into search-friendly phrasing. Fails OPEN — on LLM
  outage the raw question is used; a worse search beats a crashed pipeline.
- *Hybrid retrieval* (`infrastructure/vector_store.py`): dense vectors match
  MEANING ("money back" ~ "refund"), sparse BM25 vectors match EXACT TOKENS
  ("ND-WH-TLS"). Scores live on incomparable scales, so ranks are fused with
  RRF (Reciprocal Rank Fusion) server-side in Qdrant.
- *Reranking* (`infrastructure/reranker.py`): the funnel. Bi-encoders embed
  query and docs separately (fast, precomputable, approximate); a
  cross-encoder reads query+doc together (precise, expensive). Retrieve 20
  cheaply, rerank to 5 precisely.
- *Grounded generation* (`rag/answering.py`): the model may only use provided
  <document> blocks, must cite [n] inline, must say "I don't know" when the
  context doesn't answer. Chunks are treated as UNTRUSTED (injection hygiene).
- *Faithfulness self-check* (`rag/self_check.py` + `rag/pipeline.py`): a
  second, cheap LLM pass audits the draft against the sources; unsupported
  claims trigger ONE corrective round (re-retrieve including the claim,
  regenerate with a revision note), then the answer ships flagged
  `grounded=False`. Bounded loops — always.
- *Cost accounting* (`llm/tracking.py`): a decorator wrapping the LLM port
  accumulates tokens across every call; no component does bookkeeping.

**Why it matters in production.** Hallucination control is layered: grounding
prompt -> mandatory citations -> independent self-check -> flagged output for
human review. No single layer is reliable; the stack is. And model-tier
routing (cheap model for rewrite/check, strong for the answer) is the
difference between $0.002 and $0.02 per question at scale.

**Talking point.** "My RAG fails open on quality steps and flags-for-human on
verification failure — and every LLM call in an answer is token-accounted via
a decorator on the provider port, not bookkeeping scattered through the code."

## ⏳ The agent loop — reason → act → observe (Phase 3)
## ⏳ Multi-agent orchestration — supervisor-worker, handoffs, shared state (Phase 4)
## ⏳ MCP — servers, clients, consent model (Phase 5)
## ⏳ Memory — short-term vs long-term (Phase 6)
## ⏳ Guardrails — I/O validation, HITL, prompt-injection defense (Phase 7)
## ⏳ Observability & evaluation — traces, golden datasets, cost (Phase 8)
