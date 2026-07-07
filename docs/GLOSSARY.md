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

## Multi-agent orchestration (Phase 4)

**Supervisor-worker (hub-and-spoke).** One supervisor owns routing; specialist
agents do the work and ALWAYS return control to the hub — they never call each
other. One place to audit decisions, cap iterations, and contain failures.
Peer-to-peer agent meshes are much harder to debug; the industry converged on
hub-and-spoke for a reason. See `agents/graph.py`.

**Shared typed state.** The Pydantic object all agents read/write
(`agents/state.py`) — the INTERFACE between agents. Each node returns a
partial update; LangGraph validates and merges it. With a schema, one agent
writing garbage fails at the boundary; with loose dicts it surfaces three
agents later as a weird answer.

**Checkpointing.** Persisting the graph state after EVERY node (SQLite here,
Postgres in prod). Buys crash recovery (resume from last good step) and
pause/resume (a human approval can suspend a run for days — phase 7). The
reason a graph beats nested function calls: a call stack lives in RAM.

**Thread (thread_id).** The key of a checkpointed conversation: same id =
resume with full state; different id = fresh run. One thread per support
ticket/conversation. This is the mechanical basis of short-term memory
(phase 6).

**LLM-router vs policy-router.** Two supervisor styles. LLM-router: a model
picks the next agent each hop — flexible, but unauditable, non-deterministic,
and one paid call per hop. Policy-router (ours): the LLM's judgment is
captured ONCE as structured data (TriageDecision + confidence) and routing
over it is plain code — deterministic, testable branch by branch, free.
Intelligence at the edge, dumb auditable center. See `agents/supervisor.py`.

**Structured output (as a guardrail).** When an LLM's output drives a code
path (triage → routing), it must arrive as schema-validated data, never free
text. Our triage degrades EVERY failure (outage, bad JSON, invalid values) to
UNKNOWN/confidence 0 — which policy routes to a human. Bad classification
becomes a routable fact instead of a crash or a silent wrong turn.

**Confidence-based escalation.** The classifier reports its own confidence;
below a BUSINESS threshold (`MIN_ROUTING_CONFIDENCE`), the system refuses to
guess and hands off to a human. The design principle: a system that knows
when not to trust itself.

**Failure containment.** Specialist nodes never raise: exceptions become
entries in `state.failures`, and the supervisor routes the ticket to
escalation. One crashed agent degrades to a human handoff; it must never kill
the run.

**Direct handoff (Agents SDK style).** The alternative to a supervisor: agents
get `transfer_to_<x>` pseudo-tools and pass the LIVE conversation to each
other — same history, new persona. Less code, natural feel; but control flow
is implicit, unauditable, and two agents can ping-pong forever without a turn
budget. See `agents/handoff_demo.py` for the comparison in code.

**Nested loops (graph orchestrates agents).** Each specialist node is itself a
full phase-3 ReactAgent with its own inner tool loop: LangGraph governs the
BETWEEN-agents flow, ReAct the WITHIN-agent flow. Standard shape of
production multi-agent systems.

## MCP — Model Context Protocol (Phase 5)

**MCP.** An open protocol (Anthropic, Nov 2024; industry standard by 2026)
that decouples AI apps from their tools: a SERVER exposes capabilities in a
standard shape, and ANY MCP-capable client can use them. "USB-C for AI."
Not related to our in-process ports — MCP is a NETWORK protocol between
processes.

**The M×N problem.** Without a protocol, M apps × N tools = M·N custom
integrations; with one, each side implements the protocol once: M+N. The same
economics that gave us HTTP and LSP.

**MCP server / client / host.** Server: the process exposing tools/resources
(`mcp_servers/crm`, `mcp_servers/ticketing`). Client: the connector inside the
AI app (`mcp_clients/`). Host: the application the user actually talks to
(our agents), responsible for consent.

**Tool vs Resource (vs Prompt).** Tool = an ACTION the model may invoke
(lookup_customer, create_ticket). Resource = readable CONTEXT data addressed
by URI (`crm://customers`). Prompt = a reusable prompt template the server
offers. Tools act; resources inform.

**Streamable HTTP.** The current MCP transport (spec rev 2025-06-18): plain
HTTP requests, stateless-capable on the transport layer. Replaced the
deprecated SSE transport (common in 2024 tutorials — avoid). stdio remains
for local child-process servers.

**Tool annotations (readOnlyHint / destructiveHint).** Machine-readable
safety metadata a server attaches to each tool. Our client uses them to
decide which calls need consent — and treats MISSING annotations as
sensitive (fail closed on unknown metadata).

**Consent model.** The MCP spec requires the HOST to obtain user consent
before invoking tools, especially mutating ones. Ours lives in the CLIENT
layer (`mcp_clients/tools.py`) so no agent can forget it; a denial returns to
the model as an observation ("the user declined — don't retry"), so the agent
adapts instead of silently failing. Phase 7 upgrades the CLI prompt to a
checkpointed human-in-the-loop interrupt.

**Server-side schema validation.** Tool input schemas are generated from the
server's own type hints and enforced BEFORE the tool function runs — a
hostile or confused client can't reach tool code with malformed arguments.
First line of defense against argument injection over the protocol.

**Sync-over-async bridge.** The official SDK is async; our agent loop is
sync. The client opens a session per operation (asyncio.run) — simple and
aligned with stateless Streamable HTTP, at the cost of connection setup per
call. Phase 9's async FastAPI could hold long-lived sessions instead.

## Memory (Phase 6)

**Short-term vs long-term memory — the crisp answer.** Short-term = an EXACT
SNAPSHOT of one conversation, restored by key (thread_id -> checkpointed
graph state). Long-term = a LOSSY, SEARCHABLE DISTILLATION of everything
past, retrieved by relevance. Snapshot vs distillation is the asymmetry that
makes both necessary: without short-term there's no coherent conversation;
without long-term the customer repeats themselves every session.

**State reducer.** How LangGraph merges a node's partial update into state:
default = replace the field; `Annotated[list[X], operator.add]` = APPEND.
Our `history` uses the append reducer — that's the whole mechanism behind
multi-turn conversations accumulating across invocations of one thread.

**Turn reset.** With checkpointing, last turn's state is restored before the
new input merges in — so per-turn fields (final_answer, triage, failures,
budget) must be explicitly reset each turn or the supervisor would see the
old answer and end immediately. See `run_support_graph`.

**Profile store (exact half).** Durable key-value facts about a customer
(plan, OS, preferences) in SQLite. You want ALL facts, precisely, every time
— a SQL lookup, not a similarity search. Upsert-by-key IS the consolidation:
"plan=pro" overwrites "plan=free" instead of coexisting with it.

**Episodic memory (fuzzy half).** Past-interaction SUMMARIES in a Qdrant
collection, retrieved by semantic similarity to the current question — RAG
machinery pointed at our own history. Dense-only (no BM25 channel): recall
queries are paraphrases of situations, not error-code lookups.

**Extract → consolidate → retrieve.** The long-term memory pipeline: after
each turn a cheap LLM distills the exchange into a one-sentence episode +
durable facts (extract, `memory/writer.py`); facts upsert by key and episodes
by (thread, turn) id (consolidate); the recall node loads profile + top-k
relevant episodes into the next turn's context (retrieve). This is the
mechanism products like Mem0/Zep sell — built by hand here on purpose.

**Memory isolation.** Episodic search carries a HARD per-customer filter
enforced by the database — Dana's history is unreachable from Sam's
conversation even when semantically similar. A security property, not a
relevance heuristic, so it must not live in post-processing someone can
forget.

**Memory fails open.** Recall/write failures degrade personalization, never
availability — a support answer must not fail because the diary was down.
(Contrast with consent, which fails closed: actions vs enhancements.)

## Guardrails & human-in-the-loop (Phase 7)

**Flag, don't block.** The two-tier input policy: STRUCTURAL problems (empty,
oversized, control chars) are rejected outright; SUSPICIOUS content
(injection-looking phrases) is flagged into state for humans/audit but still
served — heuristics have false positives, and a customer quoting weird bot
output must not be locked out. See `guardrails/input_validation.py`.

**Direct vs indirect prompt injection.** Direct: the USER types "ignore your
instructions". Indirect (the dangerous one): malicious instructions hidden in
content the system reads on the user's behalf — a poisoned KB document, a
ticket body, a compromised MCP server's tool result. The model can't tell
"text to obey" from "text being processed" unless the system draws the line.

**Spotlighting.** The delimit-and-remind technique for untrusted content:
wrap it in explicit markers (`<tool_output>`, `<document>`), tell the model
its content is data, and prefix a warning when instruction-like text is
detected inside. Raises the bar; doesn't make prompts injection-proof —
structural defenses bound the damage when text-level ones fail.

**Propose vs execute.** The core agent-safety split: the LLM may PROPOSE any
action; CODE decides what executes. Schema validation, consent gates, the
refund threshold and the interrupt all live on the code side of that line —
an injected model can talk, but it cannot cross.

**interrupt() / Command(resume=...).** LangGraph's pause mechanism: a node
calls `interrupt(payload)`, the graph checkpoints and returns control with
the payload; later — same or DIFFERENT process — invoking the thread with
`Command(resume=decision)` re-enters that node with the decision as the
return value. This is why checkpointing was chosen in phase 0: a call stack
can't wait days for a human; a checkpoint can wait forever.

**Human-in-the-loop (HITL).** Irreversible actions above a business threshold
(refund > $500) hard-stop the graph until a person decides. The human's
decision — not the model's draft — becomes the customer-facing answer. See
`agents/hitl.py`.

**Fail closed (for actions).** Every ambiguous case around the irreversible
action resolves to NO: missing approval callback → denied; malformed resume
payload → denied; unannotated MCP tool → consent required. Mirror image of
memory/rewrite failing open — enhancements degrade, actions stop.

## Observability & evaluation (Phase 8)

**Span / trace.** A span = one named, timed step with attributes (model,
tokens, doc ids, error flags). Spans nest into a trace — the full tree of one
request. When an answer is wrong, the failing STEP is visible in the tree;
logs interleave and lie, traces keep causality.

**OTLP / OpenTelemetry.** The vendor-neutral telemetry standard. We emit
standard OTLP spans, so the viewer (Arize Phoenix locally) is swappable for
Datadog/Langfuse/etc. with zero instrumentation changes. Instrumentation is
forever; backends are fashion (ADR-10).

**GenAI semantic conventions.** Standardized attribute names for LLM spans
(`gen_ai.request.model`, `gen_ai.usage.input_tokens`...) so any OTel-aware
viewer renders model and token data without custom mapping.

**No-op tracing.** Until a real provider is installed, OTel spans cost ~zero
and export nowhere — so the code is ALWAYS instrumented and telemetry is
purely a deployment switch (`TRACING_ENABLED=1`). No `if tracing:` litter.

**Golden dataset.** A small, version-controlled set of question→expected
pairs ("these must keep working"). Offline evals run it on every change:
swap the embedding model or reword a prompt, and the scores catch regressions
BEFORE customers do. Complements (never replaces) the runtime self-check:
runtime catches problems per-answer, offline catches them per-change.

**hit@k.** Retrieval metric: fraction of golden questions whose expected
document appears in the top-k results. Ours gates the full funnel
(hybrid retrieve 20 → rerank 3) at hit@3 ≥ 85%.

**LLM-as-judge (offline).** Using a model to score outputs at eval time —
our faithfulness suite reuses the same FaithfulnessChecker that runs in
production, pointed at golden questions. Cheap, scalable, imperfect: judges
have false negatives, which is why thresholds are gates, not gospel.

**Cost per run.** Every CLI output ends with estimated USD (token counts ×
a per-model price table in `observability/cost.py`). Unknown models price as
the strong tier — overestimating is the safe direction for a number that
exists to catch runaway cost.

## Packaging & interface (Phase 9)

**SSE (Server-Sent Events).** One-directional HTTP streaming: the server
writes `event:`/`data:` lines down a kept-open response. Our /chat streams
node-by-node progress then the answer — agent runs take seconds, and users
tolerate latency they can SEE. (Node-level, not token-level: token streaming
would need a streaming method on the LLM port end-to-end.)

**HITL over HTTP.** When a run interrupts for approval, the /chat stream ends
with `approval_required` and the state sits checkpointed; a SEPARATE call to
/approvals/{thread} — different client, different day — resumes it. Two
stateless HTTP calls connected only by the checkpoint.

**Thin client.** The `nimbus` CLI holds zero business logic: it's an
SSE-consuming HTTP client of the API, so every demo also exercises the
production surface. One brain, two doors.

**Degrade, don't crash-loop.** The API starts even when composition fails
(missing key): /health reports "degraded" with the reason, /chat returns 503.
A container that comes up and explains itself beats one restarting forever.

**Lockfile builds.** The Docker image installs from `uv.lock` (`--frozen`):
the container runs EXACTLY the dependency set the tests ran against.
Dependencies install in their own layer before source copies in, so code
edits don't re-download the world on rebuild.
