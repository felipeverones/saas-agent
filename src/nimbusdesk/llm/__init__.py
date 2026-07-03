"""LLM access layer — the one port every LLM call in the system goes through.

WHY A SHARED PACKAGE (instead of a port inside rag/)
Phase 2's RAG pipeline and phase 3+'s agents all need completions. Defining
the contract once here means: one fake to write for tests, one place to add
cross-cutting concerns later (tracing and cost accounting in phase 8, retries,
rate limiting), and one seam to swap vendors.

WHY TWO MODEL TIERS (see infrastructure/settings.py)
Production agent systems route by task difficulty: a cheap/fast model for
high-volume mechanical steps (query rewriting, classification, self-checks)
and a strong model for user-facing reasoning. Using the strong model for
everything typically multiplies cost 5-10x for zero quality gain on the easy
steps — model-tier routing is one of the highest-leverage cost controls.

The concrete Anthropic adapter lives in infrastructure/anthropic_llm.py;
tests use tests/fakes.py::FakeLLMProvider and never touch the network.
"""
