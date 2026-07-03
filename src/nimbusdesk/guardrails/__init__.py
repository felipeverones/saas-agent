"""Guardrails layer — the safety envelope around every LLM interaction.

MENTAL MODEL: never trust, always verify — in BOTH directions.
LLMs are probabilistic; production systems are not allowed to be. Guardrails are
the deterministic shell around the probabilistic core:

- INPUT validation: schema, size limits, basic malicious-content screening
  BEFORE anything reaches an agent.
- OUTPUT validation: decisions that trigger actions (escalate? refund? priority?)
  must arrive as Pydantic-validated structured output, never free text. Free text
  is for humans; structured output is for code paths.
- HUMAN-IN-THE-LOOP (HITL): irreversible actions (refunds, customer-data changes)
  hard-stop the graph via checkpointed interrupt until a human approves. The LLM
  can PROPOSE; only a human may CONFIRM.
- PROMPT-INJECTION defense: the sneaky part is INDIRECT injection — malicious
  instructions hidden in content the system reads (a RAG document, a ticket body,
  an MCP tool result). Defense here: strict tool schemas, delimiting/marking
  untrusted content in prompts, and least-privilege tool access per agent.

WHY A DEDICATED LAYER (not per-agent checks)
Safety logic duplicated across agents WILL drift and develop gaps. One layer,
uniformly applied, is auditable — and "how do you stop your agent from doing
something stupid?" deserves a one-package answer.
"""
