"""Observability layer — traces, token accounting and cost per run.

WHY print() DOESN'T SCALE HERE
One user question fans out into: routing decision -> query rewrite -> retrieval ->
rerank -> tool calls -> generation -> self-check. When the answer is wrong, the
bug is in ONE of those steps. Logs interleave and lie; what you need is the tree.

STRUCTURED TRACING
Each step emits a SPAN (name, start/end, attributes like model, token counts,
retrieved doc ids, decision taken). Spans nest into a trace — the full tree of one
request. We use OpenTelemetry, the vendor-neutral standard, with GenAI semantic
conventions, exported to Arize Phoenix (self-hosted viewer). WHY OTel and not a
proprietary SDK: swap the backend (Phoenix -> Datadog -> Langfuse) without
touching instrumentation. Instrumentation is forever; backends are fashion.

COST TRACKING
Every LLM call records prompt/completion tokens and estimated USD. In production
agent systems, cost-per-resolved-ticket is a first-class product metric — an agent
that burns $3 of tokens on a $0.50 question is a bug even when the answer is right.
"""
