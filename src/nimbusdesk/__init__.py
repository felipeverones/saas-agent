"""NimbusDesk — a production-style AI support platform for a fictional SaaS company.

This project exists to practice, end to end, the patterns the market expects from
AI engineers in 2026: multi-agent orchestration, agentic RAG, MCP interoperability,
memory, guardrails, and observability.

ARCHITECTURE IN ONE PARAGRAPH
The codebase follows a layered ("clean") architecture. The core rule is the
DEPENDENCY RULE: source code dependencies only point INWARD, toward the domain.

    interface  ->  agents / rag / memory / guardrails  ->  domain
        \\                    |
         \\------->  infrastructure  (adapters: LLM, vector store, MCP, telemetry)

- `domain/` is the innermost circle: pure business rules, zero AI/framework imports.
- `agents/`, `rag/`, `memory/`, `guardrails/` are the application layer, organized
  by CAPABILITY rather than by technical role (see ADR-13 in docs/ARCHITECTURE.md).
- `infrastructure/` holds the adapters to the outside world (Anthropic API, Qdrant,
  SQLite, OpenTelemetry). Swapping a vendor should only ever touch this package.
- `interface/` is how humans/systems reach us: FastAPI app + interactive CLI.

WHY THIS MATTERS IN PRODUCTION
LLM apps churn dependencies faster than any other stack (frameworks, providers and
protocols all shifted between 2023 and 2026). Teams that welded business logic to a
framework rewrote their apps twice; teams that isolated it swapped adapters.
"""
