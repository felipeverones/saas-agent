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

## The agent loop — reason → act → observe (Phase 3)

**What it is.** An agent = an LLM in a loop with tools. Each iteration the
model either answers (loop exits) or emits structured tool calls; we validate
arguments, execute, and feed results back as observations. The model decides
when it's done — that's the difference between an agent and a pipeline.

**The five design rules in our implementation** (`agents/react.py`):
1. *Structured tool calling, never text parsing* — tools are declared with
   JSON Schemas generated from Pydantic models; the model returns calls as
   data. (2023's "Action: search[...]" regex parsing is obsolete — brittle
   and injectable.)
2. *Validate before execute* — every argument dict passes the tool's Pydantic
   model first; garbage bounces back as a model-readable error.
3. *Errors are observations* — failed calls return to the model
   (is_error=true) instead of raising; agents recover from errors, crashed
   processes don't.
4. *Iteration budget* — max_iterations turns "agent stuck in a loop" from an
   infinite incident into a bounded failure with a graceful message.
5. *Full step trace* — every (tool, args, observation) is recorded;
   "why did it do that?" is answerable from data.

**Where.** Loop: `agents/react.py`. Tool contract: `agents/tools.py`. The
support agent's tools + prompt: `agents/local_tools.py`,
`agents/support_agent.py`. Port extension for tool calling:
`llm/ports.py::ToolCallingLLM`, adapter in `infrastructure/anthropic_llm.py`.

**Why it matters in production.** Every agent framework (LangGraph included)
is this loop with orchestration sugar on top. Having built it raw once, you
can debug any of them — and answer "what does your framework actually do?"

**Talking point.** "My agent treats tool failures as observations and has an
explicit iteration budget — the two properties that separate a demo agent
from one you can put in front of customers."

## Multi-agent orchestration — supervisor-worker, handoffs, shared state (Phase 4)

**What it is.** A LangGraph state graph: supervisor at the hub, specialists
(triage, technical, billing, escalation) as spokes, one shared Pydantic state
as the interface between them, checkpointed to SQLite after every node.

**The flow.** START → supervisor → triage (fast model, structured
TriageDecision with confidence) → supervisor routes by POLICY (code, not an
LLM): billing→billing agent, technical/account→technical agent, low
confidence or urgent or crashed specialist→escalation. Specialists are the
phase-3 ReactAgents nested as nodes. Every spoke returns to the hub; a
supervisor-visit budget bounds the whole graph.

**Design decisions to remember.**
- *Policy-router over LLM-router*: the LLM's judgment is captured once as
  structured data; routing over it is deterministic, testable code.
  Intelligence at the edge, dumb auditable center (`agents/supervisor.py`).
- *Every failure becomes a routable fact*: triage degrades to
  UNKNOWN/confidence 0; specialist crashes become `state.failures` entries;
  both route to escalation. The customer ALWAYS gets an answer, even if the
  answer is "a human will follow up".
- *Checkpointing* (`thread_id` → persisted state per conversation) is what
  separates a graph from nested function calls: crash recovery + pause/resume.
- *Direct handoff demo* (`agents/handoff_demo.py`): the Agents-SDK-style
  alternative where agents transfer the live conversation to each other.
  Less code, implicit control flow, ping-pong risk. We built both to compare.

**Where.** State: `agents/state.py`. Policy: `agents/supervisor.py`. Nodes:
`agents/triage.py`, `agents/specialists.py`. Wiring: `agents/graph.py`.
Flows tested end-to-end: `tests/integration/test_support_flows.py`.

**Talking point.** "My supervisor is code, not an LLM — triage confidence is
structured output, and routing over it is unit-tested branch by branch. The
LLMs decide WHAT things are; the system decides WHERE they go."

## MCP — servers, clients, consent model (Phase 5)

**What it is.** Two standalone MCP servers (CRM, ticketing) built on the
official SDK, speaking Streamable HTTP; a sync client layer that discovers
their tools at runtime and adapts each one into the same `ToolLike` port the
agents already use. The phase-3 local customer lookup and the phase-5 remote
one are interchangeable — `--mcp` flips between them with zero agent changes.

**The load-bearing ideas.**
- *Decoupling is real, not rhetorical*: the servers import NOTHING from
  nimbusdesk (they simulate other teams' systems), and the agent can't tell
  local from remote tools. `load_remote_tools()` gains whatever the server
  ships — today's tools and future ones — with no code changes (the M×N
  payoff).
- *Consent gate in the client layer*: read-only tools (per server
  annotations) run freely; everything else requires explicit user approval,
  and a denial is returned to the MODEL as an observation so it adapts.
  Missing annotations = sensitive (fail closed).
- *Schema validation at the protocol boundary*: FastMCP generates schemas
  from the server's type hints and enforces them before tool code runs —
  malformed/hostile arguments bounce at the door.
- *Transport currency*: Streamable HTTP (2025-06-18 spec revision); the SSE
  transport you'll still see in old tutorials was deprecated in 2025.

**Where.** Servers: `mcp_servers/crm/`, `mcp_servers/ticketing/`. Client +
consent: `mcp_clients/client.py`, `mcp_clients/tools.py`. Port they satisfy:
`agents/tools.py::ToolLike`. Full-protocol tests over memory streams:
`tests/integration/test_mcp.py`.

**Why it matters in production.** Tool integrations are where AI platforms
ossify: every hardwired integration is future migration debt. MCP turns tools
into a marketplace — and its security surface (consent, annotations, schema
validation, treating tool RESULTS as untrusted input) is now standard
interview material.

**Talking point.** "My agents discover their CRM tools over MCP at runtime;
the consent gate lives in the client layer so no agent can forget it, and
unannotated tools are treated as sensitive — fail closed on unknown metadata."

## Memory — short-term vs long-term (Phase 6)

**What it is.** Two memories with different physics:
- *Short-term* (`history` in `agents/state.py` + LangGraph checkpointing):
  the exact state of ONE conversation, keyed by thread_id, restored on every
  invocation. An append REDUCER (`Annotated[list, operator.add]`) makes the
  history accumulate while per-turn fields get reset (`run_support_graph`).
- *Long-term* (`memory/`): a lossy, searchable digest across ALL sessions.
  Structured facts → SQLite (`profile_store.py`, exact lookup, upsert-by-key
  consolidation); episode summaries → Qdrant (`episodic.py`, similarity
  lookup, hard per-customer isolation filter). A cheap LLM extracts both
  after every turn (`writer.py`); the graph's recall node injects them at
  the start of the next one.

**The interview answer, verbatim.** "Short-term memory is a snapshot —
exact, per conversation, restored by key. Long-term is a distillation —
lossy, cross-session, retrieved by relevance. You need both because a
snapshot can't scale beyond one thread and a distillation can't reconstruct
a live conversation."

**Design decisions worth remembering.**
- Distill at write time, not read time: raw transcripts retrieve poorly and
  make every recall a re-reading job.
- Exact facts do NOT go in the vector store (fuzzy where you want precision
  is over-engineering); fuzzy episodes do NOT go in SQL (you don't know the
  key you'll need).
- Memory isolation is enforced in the database (Qdrant filter), not in
  application code someone can forget.
- Memory fails OPEN everywhere: it's an enhancement, never a dependency.

**Where.** `memory/` (stores, writer, service facade), recall/finalize nodes
in `agents/graph.py`, flows in `tests/integration/test_memory_flows.py`.
Demo: `make chat EMAIL=dana@acme.io` across two different THREAD values.

**Talking point.** "My long-term memory is extract→consolidate→retrieve built
by hand: facts upsert by key so contradictions resolve, episodes upsert by
(thread, turn) so re-processing is idempotent, and per-customer isolation is
a database filter, not an application promise."

## Guardrails — I/O validation, HITL, prompt-injection defense (Phase 7)

**What it is.** The deterministic shell around the probabilistic core, in
four layers:
1. *Input gate* (`guardrails/input_validation.py` + guard_input node):
   structural problems rejected; injection-looking phrases flagged into
   state, never silently blocked (flag-don't-block).
2. *Untrusted-content defense* (`guardrails/injection.py`): every tool
   result is delimited, size-capped, and warning-prefixed when it contains
   instruction-like text (spotlighting) — the INDIRECT injection vector
   (poisoned docs, hostile tool output) is the one that matters.
3. *Output validation*: decisions that drive code paths arrive as
   schema-validated structured output (triage since phase 4); free text is
   for humans only.
4. *Human-in-the-loop* (`agents/hitl.py`): refunds over $500 (a DOMAIN rule,
   `RefundRequest.requires_human_approval`) pause the graph via
   `interrupt()`; the checkpoint waits — hours, days, another process — and
   `Command(resume=...)` continues exactly where it stopped. The human's
   decision becomes the answer; the model's optimistic draft is discarded.

**The design principle tying it together: propose vs execute.** The LLM may
propose anything; code decides what runs. And failure directions are chosen
per kind: quality steps fail OPEN (rewrite, memory), actions fail CLOSED
(consent, refunds — no human available means denied, never executed).

**Where.** `guardrails/`, `agents/hitl.py`,
`agents/local_tools.py::RequestRefundTool`, routing priority in
`agents/supervisor.py`, interrupt loop in
`agents/graph.py::run_support_graph`. Flows:
`tests/integration/test_hitl_flow.py` — including resume-from-a-fresh-
process, the test that proves the phase-0 architecture bet.

**Talking point.** "My refund flow is propose-vs-execute: the model can only
register a request; above the business threshold the graph checkpoints and
interrupts until a human resumes it — from any process, any time later. And
every ambiguity around money resolves to 'denied': actions fail closed."

## Observability & evaluation — traces, golden datasets, cost (Phase 8)

**What it is.** Three lenses on a system you can't fully predict:
1. *Traces* (`observability/tracing.py`): every request produces a span tree
   — graph.<node> → agent.loop → llm.complete / tool.<name> → rag.retrieve /
   rerank / self_check — exported as standard OTLP to Arize Phoenix
   (`make up`, http://localhost:6006). The no-op trick: code is always
   instrumented; TRACING_ENABLED=1 decides whether telemetry flows.
2. *Cost* (`observability/cost.py`): every CLI run ends with estimated USD
   from token counts × a per-model price table. Cost-per-resolved-ticket is
   a product metric, not an afterthought.
3. *Offline evals* (`evals/run_eval.py` + golden_dataset/): hit@3 over the
   real retrieval funnel (free, 17 cases), routing accuracy over real
   triage+policy (12 cases, needs key), faithfulness of full pipeline
   answers judged by our own checker (8 cases, needs key). Thresholds gate
   CI via exit code.

**The distinction that matters in interviews: runtime checks vs offline
evals.** The runtime self-check catches problems per-ANSWER in production;
the golden dataset catches regressions per-CHANGE in CI. You need both:
runtime can't tell you the new embedding model dropped recall 10%, and no
offline suite covers every live question.

**Where.** `observability/` (tracing, TracingLLM decorator, cost),
instrumentation inline in react/retrieval/pipeline/graph, evals under
`evals/`. Span-tree shape pinned by `tests/unit/test_observability.py`
against an in-memory exporter.

**Talking point.** "My instrumentation is vendor-neutral OTLP with GenAI
semantic conventions — Phoenix today, Datadog tomorrow, zero code changes.
And my eval suite reuses the production components as their own judges, so
eval and runtime can't drift apart."

