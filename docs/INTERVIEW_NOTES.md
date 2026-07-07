# INTERVIEW_NOTES.md — likely questions & guided answers

Answers use a compact STAR shape (Situation/Task → Action → Result) plus the
trade-off you should volunteer. Grows at the end of each phase.

---

## Q: "Walk me through the architecture. Why layered?"

**S/T.** Multi-agent support platform: supervisor routing to specialists, agentic
RAG, MCP tool servers, dual memory, guardrails, OTel tracing.
**A.** Layered core with the dependency rule enforced by a CI test: `domain/` has
zero AI imports; agents depend on an `LLMProvider` port, vendors live in
`infrastructure/` adapters. Capabilities (RAG, memory, guardrails) are vertical
packages so features have one home.
**R.** Whole test suite runs against a fake LLM — zero tokens, no keys in CI;
provider migration is one adapter.
**Trade-off to volunteer.** Ports/adapters add indirection a demo doesn't need —
it pays off at the first vendor change or the first test suite, whichever comes
first.

## Q: "Why LangGraph and not CrewAI / plain function calls?"

**A.** I needed three properties CrewAI abstracts away and a while-loop can't
give: (1) typed shared state checkpointed after every node → crash recovery;
(2) `interrupt()` on that checkpoint → human-in-the-loop that can wait days;
(3) explicit transitions → every routing decision is a span I can audit.
**Trade-off.** More boilerplate than CrewAI; I accepted it because control and
auditability were requirements, not nice-to-haves. I also built one direct-handoff
flow (Agents SDK style) to know the alternative first-hand: handoffs are less
code, graphs are more governable.

## Q: "Why did you build your own long-term memory instead of Mem0/Zep?"

**A.** Deliberate learning choice: I wanted to own the extract → consolidate →
retrieve pipeline. Structured facts go to SQLite (exact lookup), episodic
summaries to Qdrant (semantic lookup). In a product company I'd evaluate Mem0/Zep
first — buying is usually right when memory isn't your differentiator.

## Q: "How did you decide your chunking strategy and chunk size?"

**S/T.** Markdown knowledge base; retrieval quality depends on chunks that are
topically coherent and self-explanatory.
**A.** Structure-aware chunking: split on headings first (the author's own
topic segmentation), size-split only oversized sections on paragraph
boundaries with overlap, and carry the heading trail into each chunk. I embed
the chunk WITH its title/section prefix but store the raw text, so embeddings
get context and citations stay verbatim.
**R.** Integration tests asking questions in user language ("customer wants
money back after 3 weeks") hit the right article top-1 with no keyword overlap.
**Trade-off.** Fixed-size chunking is simpler and works on unstructured text;
I'd fall back to it (or sentence-window chunking) for OCR dumps and chat logs
where headings don't exist. Chunk size (~1200 chars) is a recall/precision
dial — the honest answer is you tune it against an eval set, which is phase 8.

## Q: "How do you test a RAG system without burning API calls?"

**A.** Two tiers. Unit: everything behind ports — a hash-based FakeEmbedder
and an in-memory index prove mechanics (idempotent upserts, ranking, k) in
milliseconds, offline. Integration: real local embedding model + Qdrant's
in-process mode (`QdrantClient(":memory:")`) prove semantic quality against
the real corpus — still free, because embeddings run locally on CPU.
**Trade-off.** Local small model ≠ production embedding quality; the golden
eval set (phase 8) is what catches model-swap regressions.

## Q: "How would you avoid hallucination in a RAG system?"

**A.** Layered, because no single defense is reliable:
(1) retrieval quality first — hybrid dense+BM25 with reranking, since most
"hallucinations" are actually the model compensating for bad retrieval;
(2) grounding contract in the prompt — answer only from provided documents,
cite [n] per claim, "I don't know" is a valid output;
(3) independent faithfulness check — a second cheap LLM pass audits the draft
against sources and lists unsupported claims;
(4) one bounded corrective round (re-retrieve + regenerate with the failed
claims), then ship flagged `grounded=False` for human review instead of
looping.
**R.** In my project this is `rag/pipeline.py`: the flag reaches the caller,
so the UI/agent must warn or escalate — the system never silently asserts an
unverified answer.
**Trade-off.** The check adds one LLM call (~30% latency); mitigated by using
the fast model tier. And LLM-as-judge has false negatives — which is why
phase 8 adds offline eval on a golden dataset instead of trusting the runtime
check alone.

## Q: "Why hybrid search? Isn't vector search enough?"

**A.** Dense embeddings encode meaning but blur rare exact tokens — error
codes, SKUs, function names. BM25 nails exact tokens but misses paraphrase
("money back" vs "refund"). Support traffic is a mix, so I run both channels
and fuse RANKS with RRF — scores are on incomparable scales, ranks aren't.
**R.** My integration suite has both query classes; dense-only failed the
"error ND-WH-TLS" class, hybrid passes all.
**Trade-off.** Two embeddings per query + a fusion step; Qdrant does fusion
server-side so latency impact is minimal. BM25 needs no training but is
vocabulary-exact — typo tolerance would need a third trick (not needed here).

## Q: "How would you debug an agent that got stuck in a loop?"

**S/T.** Looping is THE classic agent failure: a confusing tool result makes
the model retry the same call forever.
**A.** Prevention first: my loop has a hard `max_iterations` budget — on
exhaustion it returns a graceful "escalating to a human" answer flagged
`hit_iteration_limit=True`, so loops are bounded, visible failures, never
hung processes. Diagnosis second: every run records a full step trace
(tool, arguments, observation per iteration), so I can replay exactly what
the model saw before each repeated decision — the culprit is almost always an
ambiguous observation or a tool description that doesn't match behavior.
**R.** In tests I script a model that always requests tools and assert the
loop exits at the budget with the fallback answer (`test_react.py`).
**Trade-off.** A tight budget truncates legitimately long tasks; the right
value is task-dependent and belongs in evals (phase 8), not in a constant you
guess once.

## Q: "How do you keep an agent's tool calls safe?"

**A.** Three layers in phase 3, more in phase 7: (1) the model never executes
anything — it emits a structured request, my code executes; (2) every tool's
arguments are validated against a Pydantic schema BEFORE execution, and the
schema shown to the model is generated from that same model, so docs and
validation can't drift; (3) failures return as error observations the model
can read, while internal stack traces stay in MY logs — models don't need
implementation details, and leaking them helps injection attacks.

## Q: "Supervisor pattern vs direct handoffs — when would you use each?"

**S/T.** I implemented BOTH in the same project to compare honestly.
**A.** Supervisor graph (LangGraph): hub-and-spoke, every specialist returns
to the hub, routing is explicit and checkpointed. Direct handoff (Agents SDK
style): agents carry `transfer_to_<x>` tools and pass the live conversation
to each other — same history, new persona.
**R.** Handoffs took ~120 lines and read naturally, but control flow is
implicit (who reaches whom is buried in tool lists) and two agents can
ping-pong forever without a turn budget — I have a test demonstrating exactly
that failure. The graph costs more ceremony and buys auditability,
per-step checkpointing, and one place to enforce budgets.
**Rule of thumb.** Handoffs for small conversational systems (2-3 personas);
state graph when you need audit trails, crash recovery, or human-in-the-loop.

## Q: "Why is your supervisor plain code instead of an LLM?"

**A.** I separate judgment from routing. The LLM judges ONCE — triage returns
structured data (category, priority, confidence) validated against a schema.
Routing over that data is a pure function: deterministic, free, and
unit-tested branch by branch (low confidence → human, urgent → human,
billing → billing agent...). An LLM-router re-judges at every hop: a paid
call each time, and "why did it route there?" has no answer you can test.
**Trade-off.** A policy router only handles anticipated lanes. For an
open-ended assistant with dozens of dynamic skills, I'd flip to an LLM router
— and accept the auditability cost consciously.

## Q: "What happens when one of your agents fails mid-ticket?"

**A.** Failures are state, not exceptions. Specialist nodes catch everything
and append to `state.failures`; the supervisor's policy routes the ticket to
escalation, which produces a customer-facing handoff message
deterministically (no LLM in the escalation path on purpose — when the system
is already broken, the last thing you want is another call that can fail).
The invariant my flow tests pin: every run ends with an answer, even when a
specialist crashed — a crashed agent degrades the ticket, never kills it.

## Q: "Why MCP instead of just defining tools in code?"

**S/T.** Phase 3 had in-process tools; phase 5 moved the CRM/ticketing ones
behind MCP servers.
**A.** Three reasons. Economics: M apps × N tools becomes M+N once both sides
speak one protocol — my agent gained the ticketing server's tools via runtime
discovery, zero integration code. Organizational: the CRM server simulates a
system another team owns; they ship tools on their schedule, my agent picks
them up. Security: the protocol carries machine-readable safety metadata
(read-only/destructive annotations) that my client turns into a consent gate.
**R.** The swap proof: local and MCP customer lookup are interchangeable
behind the same `ToolLike` port — one flag flips them, agents unchanged.
**Trade-off.** A protocol hop adds latency and an operational dependency
(another process to run/monitor). For tools that live and die with your app,
in-process is fine; MCP pays off at the integration boundary.

## Q: "How do you secure MCP tool calls?"

**A.** Defense in depth, both directions. Inbound to the server: input
schemas are generated from the server's type hints and enforced BEFORE tool
code runs — malformed or injected arguments bounce at the protocol boundary.
On the client/host side: the MCP consent model — read-only tools (per server
annotations) run freely, mutating ones require explicit user approval, and
missing annotations are treated as sensitive (fail closed). A denial goes
back to the model as an observation ("user declined — don't retry") so the
agent adapts gracefully. And tool RESULTS are treated as untrusted input in
prompts (indirect injection vector), which phase 7 hardens further.

## Q: "Explain short-term vs long-term memory in your agent system."

**A.** Different physics. Short-term is an exact SNAPSHOT of one
conversation: LangGraph checkpoints the typed state after every node, keyed
by thread_id; an append reducer accumulates the message history while
per-turn fields reset. Long-term is a lossy, searchable DISTILLATION across
sessions: after each turn a cheap LLM extracts a one-sentence episode
(→ Qdrant, similarity search) and durable facts (→ SQLite, exact lookup with
upsert-by-key consolidation); a recall node injects both at the start of the
next conversation.
**R.** Demo: chat with a customer on thread A, close everything, open thread
B — the system greets them knowing their OS and past issue. Tests pin it:
history accumulation, cross-session recall, and per-customer isolation.
**Trade-offs to volunteer.** Distillation loses detail by design (raw
transcripts retrieve poorly); extraction costs one cheap LLM call per turn;
and I built it by hand for learning — in a product I'd evaluate Mem0/Zep
first unless memory is the differentiator.

## Q: "How do you prevent one customer's data leaking into another's session?"

**A.** For memory: the episodic search carries a mandatory per-customer
filter EXECUTED BY THE DATABASE (Qdrant payload filter) — isolation is a
storage-layer property, not an application-code promise someone forgets in a
refactor. Profile facts are keyed by email in SQL. I have a test that stores
one customer's episode and asserts a semantically identical query from
another customer returns nothing.

## Q: "How do you defend against prompt injection?"

**A.** By assuming text-level defenses WILL eventually fail and layering
structural ones that bound the damage. Text level: input flagging (structural
rejects, suspicious flags — never block on weak heuristics), and spotlighting
for the indirect vector — every RAG chunk and tool result is delimited,
size-capped, and warning-prefixed when it contains instruction-like text.
Structural level, where the real safety lives: the model only ever PROPOSES;
tool arguments are schema-validated before execution; mutating MCP tools need
explicit user consent; and irreversible actions above a business threshold
hard-stop for human approval regardless of what the model says. An injected
model can talk; it cannot move money.
**Trade-off.** Layers add latency and code; the alternative — trusting a
phrase blocklist — is how the publicized incidents happened.

## Q: "How does your human-in-the-loop actually work under the hood?"

**S/T.** Refunds over $500 require a human by business rule.
**A.** The refund tool never executes above the limit — it registers a
typed RefundRequest that the billing node lifts into graph state. The
supervisor routes to an approval node that calls LangGraph's `interrupt()`:
the state checkpoints, the process can exit, and the payload (action, amount,
reason) surfaces to whatever operator UI. Resuming with
`Command(resume={"approved": ...})` re-enters the node with the decision;
the human's verdict — not the model's draft — becomes the customer answer.
**R.** My test suite proves the pause survives the process: interrupt on one
graph instance, resume on a freshly built one sharing only the checkpointer.
**Trade-off to volunteer.** Every interrupt is support latency; the
threshold is a business dial (auto-approve limit vs risk), not an
engineering constant.

## Q: "How would you know if a change made your RAG/agents worse?"

**S/T.** LLM systems fail silently: nothing crashes, quality just drops.
**A.** Two complementary nets. Offline: a version-controlled golden dataset
(retrieval hit@3, routing accuracy, faithfulness rate) runs on every change
with thresholds gating CI — swapping the embedding model or rewording the
triage prompt shows up as a score delta before deploy. Runtime: OTel traces
per request (every node, tool call, LLM call with token usage) exported to
Phoenix, plus a per-answer faithfulness check whose failures ship flagged.
**R.** My retrieval suite runs free (local embeddings + in-process Qdrant),
so it's viable on every PR; LLM-dependent suites run when a key is present.
**Trade-off.** Golden sets rot as the product evolves — they need the same
care as any test suite; and LLM-as-judge metrics have variance, so I treat
thresholds as tripwires, not precision instruments.

## Q: "Walk me through debugging a wrong answer in production."

**A.** Open the trace, not the logs. One request = one span tree:
guard_input → recall → supervisor → triage → specialist's agent.loop →
each tool call with arguments and error flags → rag.retrieve with the doc
ids it surfaced → rerank → generate → self_check verdict. The wrong answer
is almost always attributable to one visible step: retrieval surfaced the
wrong docs (fix chunking/query), the reranker buried the right one, a tool
errored and the model improvised, or generation ignored the context (the
self-check verdict tells you). Token counts per span also expose the cost
shape of the failure. Logs interleave concurrent requests and hide
causality; the span tree IS the causality.

---

⏳ To be added per phase: hallucination prevention in RAG (2), debugging a looping
agent (4), MCP security model (5), prompt-injection defenses (7), how evals catch
regressions (8).
