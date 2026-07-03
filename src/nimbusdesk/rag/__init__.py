"""RAG layer — Retrieval-Augmented Generation, the agentic (production) version.

THE CORE IDEA
LLMs only know their training data. RAG fixes that: at question time we RETRIEVE
the relevant snippets from our own documents and paste them into the prompt, so
the model answers from evidence instead of from memory. It is the standard cure
for hallucination and stale knowledge — and it's cheaper than fine-tuning.

PIPELINE (built in phases 1-2)
  ingestion:  documents -> chunks -> embeddings -> vector store (Qdrant)
  query:      rewrite -> hybrid retrieval -> rerank -> generate -> self-check

WHAT MAKES IT "AGENTIC" (vs the naive 2023 tutorial version)
Naive RAG is a fixed pipe: embed question, fetch top-4, answer. Production RAG
gives the LLM control points:
- QUERY REWRITING: user phrasing is rarely good search phrasing.
- HYBRID RETRIEVAL: dense vectors catch meaning ("app is slow" ~ "performance
  degradation"); sparse/BM25 catches exact tokens (error codes, product names).
  Either alone misses real queries; scores are fused server-side in Qdrant.
- RERANKING: retrieval optimizes recall (find 20 candidates fast); a cross-encoder
  then optimizes precision (score each candidate against the query, keep the best).
- SELF-CHECK: before answering, verify every claim is supported by the retrieved
  text; if not, retrieve again or refuse. Groundedness > eloquence.
- CITATIONS: every answer carries (document, snippet) provenance. An unsourced
  answer from a support bot is a liability, not a feature.
"""
