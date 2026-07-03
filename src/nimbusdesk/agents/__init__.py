"""Agents layer — multi-agent orchestration (supervisor-worker pattern, LangGraph).

WHAT IS AN "AGENT" (the 2026 working definition)
An LLM running in a loop: it reasons about a goal, chooses a TOOL (an action),
observes the result, and repeats until done. One reason->act->observe cycle is
often called a ReAct step. Phase 3 builds a single agent so this loop is concrete
before we compose several of them.

THE SUPERVISOR-WORKER PATTERN (phase 4)
One supervisor agent owns routing: it reads the shared state and decides which
specialist (triage, technical, billing, escalation) acts next. Specialists never
call each other directly — they return control to the supervisor.
WHY: a central router gives one place to audit decisions, cap iterations
(anti-infinite-loop) and contain failures. Peer-to-peer agent meshes are much
harder to debug; that's why the industry converged on hub-and-spoke topologies.

WHY A STATE GRAPH (LangGraph) AND NOT A WHILE-LOOP
The flow is modeled as a graph: nodes = agents/steps, edges = allowed transitions,
plus one shared, TYPED state object. After every node the state is checkpointed
(persisted). That buys three things a plain loop can't offer:
1. Pause/resume: human-in-the-loop approval can suspend a run for days (phase 7).
2. Auditability: every transition is inspectable and traceable (phase 8).
3. Fault tolerance: a crash resumes from the last checkpoint, not from zero.

`handoff_demo/` implements the ALTERNATIVE pattern — direct agent-to-agent handoff
(OpenAI Agents SDK style, where routing is itself a tool call) — so both models
can be compared hands-on. Trade-off in one line: handoffs are simpler to write,
graphs are easier to audit and control.
"""
