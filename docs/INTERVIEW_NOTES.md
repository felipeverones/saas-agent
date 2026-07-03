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

---

⏳ To be added per phase: hallucination prevention in RAG (2), debugging a looping
agent (4), MCP security model (5), prompt-injection defenses (7), how evals catch
regressions (8).
