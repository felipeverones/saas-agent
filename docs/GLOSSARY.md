# Glossary — every concept in this project, in plain words

Organized by theme (roughly the order the phases introduce them). Each entry:
what it is, why we use it, where to see it in the code. Grows every phase.

---

## Architecture (Phase 0)

**Layered (clean) architecture.** Code organized in concentric circles:
`domain/` (business rules) in the center, `infrastructure/` (vendors) at the
edge, `interface/` as the door. The point: the parts that change yearly (AI
frameworks) must not contaminate the parts that barely change (business rules).

**Dependency rule.** Imports only point INWARD (toward domain). `domain/`
imports nothing from the rest of the app — so no tech swap can ever break a
business rule. Enforced by `tests/unit/test_architecture.py`.

**Port.** An interface declared by the code that NEEDS something ("I need
something with an `embed_query` method") — a wall socket. Plain Python
`typing.Protocol`, same process, ordinary function calls. Not related to MCP.
See `rag/ports.py`, `llm/ports.py`.

**Adapter.** A concrete implementation of a port, wrapping a real vendor — the
plug that fits the socket. All adapters live in `infrastructure/`. Swap vendor
= write one new adapter, touch nothing else.

**Structural typing (Protocol).** A class satisfies a Protocol just by having
the right methods — no inheritance needed. This is why test fakes
(`tests/fakes.py`) don't import the real adapters at all.

**Composition root.** The ONE place where ports meet adapters — where real
objects are constructed and injected. Ours: `rag/__main__.py`. Everything
above it is vendor-agnostic.

**Architectural fitness function.** An automated test that enforces an
architecture rule (ours: "domain imports no vendors"). Conventions decay under
deadline pressure; CI doesn't.

**ADR (Architecture Decision Record).** A short log entry per non-obvious
choice: context → decision → rejected alternative → consequences. See
`docs/ARCHITECTURE.md`. The rejected alternative is the interview gold.

**Idempotency.** Running an operation N times has the same effect as once.
Our ingestion is idempotent: re-running `make ingest` never duplicates data.
Production pipelines are re-run constantly (deploys, retries, cron) — non-
idempotent pipelines corrupt themselves.

**UUID5 (vs UUID4).** UUID4 is random (new id every call). UUID5 is a HASH of
a name: same input → same id, forever, on any machine. We derive chunk ids
from `doc_id + position` (`rag/chunking.py::make_chunk_id`), which is what
makes re-ingestion idempotent.

**Upsert.** "Update or insert": write a record under an id; if the id exists,
overwrite. UUID5 + upsert = re-ingestion overwrites instead of duplicating.

## RAG fundamentals (Phase 1)

**RAG (Retrieval-Augmented Generation).** Answer from retrieved evidence, not
from model memory: Retrieve relevant snippets → Augment the prompt with them →
Generate an answer grounded in them. Cure for hallucination and stale
knowledge; cheaper than fine-tuning.

**Embedding.** A model maps text to a vector (here: 384 numbers) where similar
meanings land close together. "money back" and "refund" become neighbors with
zero shared words. See `infrastructure/embeddings.py`.

**Dense vector.** The embedding above — every dimension has a value; encodes
MEANING. Great at paraphrase, blurry on rare exact tokens (error codes).

**Chunk / chunking.** The retrieval unit: a slice of a document. We split on
markdown headings first (the author's own topic boundaries), size-split only
oversized sections. Too small = loses context; too big = blurry vector.
See `rag/chunking.py`.

**Overlap.** When force-splitting a long section, each piece starts with the
previous piece's last paragraph, so no sentence loses its neighbor at an
arbitrary cut point.

**Contextual enrichment (lite).** We EMBED "title — section ⏎ text" but STORE
raw text: the vector regains the topic signal the chunk lost when cut; the
citation stays verbatim. (The heavyweight version — an LLM writing a bespoke
context line per chunk — is "contextual retrieval".)

**Asymmetric retrieval.** Retrieval models are trained to place short
QUESTIONS near the long PASSAGES that answer them — two different encoders
paths (`embed_query` vs `embed_passages`). Using one `embed()` for both is a
classic silent-quality bug.

**Vector store.** A database indexing vectors for fast nearest-neighbor
search. Ours: Qdrant (`infrastructure/vector_store.py`), chosen for native
hybrid search (ADR-03).

**Cosine similarity.** The "how close are two vectors" metric our embedding
model was trained for. The distance metric must match the model's training.

## Agentic RAG (Phase 2)

**Sparse vector.** One dimension per vocabulary term, almost all zeros —
stored as (index, value) pairs. Encodes WHICH EXACT TOKENS appear. This is the
data shape of lexical search.

**BM25.** The classic lexical ranking formula (the heart of search engines for
decades): scores documents by term frequency, weighted by term rarity, with
length normalization. Nails exact identifiers ("ND-WH-TLS") that embeddings
blur.

**IDF (inverse document frequency).** The "rare term = strong signal" weight
inside BM25: matching "the" means nothing, matching "ND-WH-TLS" means
everything. Qdrant applies it server-side (`Modifier.IDF`).

**Hybrid search.** Run dense (meaning) AND sparse (exact tokens) retrieval,
then fuse the rankings. Real user traffic mixes both phrasing styles; either
channel alone silently fails half of it.

**RRF (Reciprocal Rank Fusion).** How the two channels merge: dense scores
(~0..1) and BM25 scores (unbounded) are incomparable, so RRF ignores scores
and sums 1/(rank + constant) from each list. Scale-free, robust, industry
default.

**Bi-encoder.** Embeds query and document SEPARATELY → document vectors are
precomputed once, search is a fast lookup. Cheap, scalable, approximate. This
is what "vector search" is.

**Cross-encoder.** Reads query + document TOGETHER as one input and scores
their actual interaction. Far more precise, far too slow for a whole corpus
(nothing precomputable). See `infrastructure/reranker.py`.

**Reranking / the funnel.** The production pattern combining the two:
retrieve ~20 candidates with the cheap bi-encoder (optimize RECALL), rerank to
top-5 with the cross-encoder (optimize PRECISION). Retrieval mistakes are
unrecoverable downstream — the generator can only cite what retrieval surfaced.

**Query rewriting.** A cheap LLM turns human phrasing ("hey, my boss wants…")
into search phrasing before retrieval. See `rag/rewrite.py`.

**Grounding / grounded generation.** The model may answer ONLY from the
provided documents; its training knowledge is treated as unreliable for our
specifics; "I don't know" is a valid output. See `rag/answering.py`.

**Citation.** Machine-checkable provenance: every claim carries an inline [n]
marker mapping to (document, section, snippet). An unsourced support answer is
a liability. See `domain/knowledge.py::Citation`.

**Faithfulness / self-check.** A second, cheap LLM pass audits the draft
against the sources and lists unsupported claims ("LLM-as-judge, pointed at
ourselves"). Verification is a different task than generation — a fresh
narrow-rubric pass beats "be careful" in the generation prompt. See
`rag/self_check.py`.

**Bounded correction loop.** Unsupported claims trigger ONE retry
(re-retrieve including the claim, regenerate with a revision note), then the
answer ships flagged `grounded=False`. Every loop needs an explicit budget —
unbounded self-correction is how agent systems melt into infinite loops.

**Fail open vs fail closed.** When a safety/quality step itself errors: fail
OPEN = proceed anyway (right for quality optimizations — a worse search beats
a crashed pipeline); fail CLOSED = block (right for actions with consequences,
phase 7). Knowing which is which is the actual engineering skill.

**Graceful degradation.** The system loses quality, not availability, when a
dependency fails — e.g. rewrite falls back to the raw question on LLM outage.

**Model-tier routing.** Cheap/fast model for mechanical steps (rewrite,
self-check, classification), strong model for user-facing reasoning. One of
the highest-leverage cost controls in LLM systems (5-10x on the easy steps).

**Decorator pattern (usage tracking).** `llm/tracking.py::UsageTracker` wraps
any LLMProvider, satisfies the same Protocol, and accumulates token counts on
the side — cost accounting in ONE place instead of bookkeeping in every
component. Same hook where tracing attaches in phase 8.

**Token / cost accounting.** Every answer reports total input/output tokens
across ALL its LLM calls. Cost-per-resolved-ticket is a product metric: a $3
answer to a $0.50 question is a bug even when correct.

## Agents (Phase 3)

**Agent.** An LLM running in a loop with tools: it decides, per iteration,
whether to answer or to act. The model itself chooses when it's done — that
autonomy is what separates an agent from a fixed pipeline.

**ReAct (Reason + Act).** The canonical agent loop: reason about the goal →
act (call a tool) → observe the result → repeat. See `agents/react.py`.

**Tool.** A capability the agent can request: (name, when-to-use description,
JSON Schema of arguments). The model only EMITS a structured request; our
code validates and executes it. That split is the security boundary of every
agent system. See `agents/tools.py`.

**Tool calling / function calling.** The provider-level feature where models
return structured tool-call blocks (id, name, typed arguments) instead of
prose like "Action: search[query]". The 2023 regex-parsing approach was
brittle and injectable; structured calls are data, not text to parse.

**Observation.** Whatever comes back from a tool, fed into the next loop
iteration — including ERRORS. "Errors are observations, not exceptions": a
failed call goes back to the model (is_error=true) so it can retry with fixed
arguments, pick another tool, or tell the user. One flaky tool must not kill
the conversation.

**Iteration budget (max_iterations).** The hard cap on loop cycles. An agent
that can loop WILL eventually loop; the budget turns an infinite incident into
a bounded, observable failure with a graceful fallback message. Same principle
as the RAG correction-round cap.

**Schema-validated tool inputs.** Every tool's arguments pass through its
Pydantic model before execution — malformed or hostile arguments bounce back
as model-readable error observations. First line of defense against argument
injection (phase 7 adds more).

**System prompt as policy.** Tool descriptions say what each tool does; the
system prompt says how THIS agent combines them (check status before
troubleshooting, search KB before quoting policy). See
`agents/support_agent.py`.

**Step trace.** The recorded list of (tool, arguments, observation) per run —
how you answer "why did the agent do that?" after the fact. Becomes
OpenTelemetry spans in phase 8.

**RAG-as-a-tool.** The phase-2 retrieval stack wrapped as just another tool
the agent may choose (`search_knowledge_base`). Modern framing: retrieval is
a capability, not a mandatory pipeline stage.

**Agent harness.** The production shell AROUND the loop: context management,
memory persistence, checkpointing/resume, tool sandboxing and permissions,
sub-agents, tracing, cost control. The loop is the engine; the harness is the
car. Nothing replaced the loop between 2023 and 2026 — what evolved is this
packaging. Phases 4-8 of this project build a harness piece by piece
(orchestration+checkpointing → memory → guardrails → observability).

**Context window.** The finite amount of text the model can see per call. A
raw loop's only "memory" is the conversation transcript inside it — which
overflows on long conversations and dies with the process. Every real memory
technique is about managing what lives OUTSIDE the window and deciding what
gets injected back in.

**Context engineering.** The 2026 discipline that displaced "prompt
engineering" as the focus: deciding what enters the finite context window at
each iteration — which retrieved chunks, which memories, which summarized
history — rather than wordsmithing one perfect instruction.

**Compaction / summarization.** Replacing older conversation turns with an
LLM-written summary to keep long sessions inside the context window, trading
detail for continuity.
