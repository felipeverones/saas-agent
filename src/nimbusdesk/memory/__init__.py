"""Memory layer — what the system remembers, and for how long.

TWO KINDS OF MEMORY (classic interview question — know this cold)

SHORT-TERM (thread-scoped): the state of ONE conversation — messages so far,
current agent, pending approvals. Implemented via LangGraph checkpointing: the
graph state is persisted (SQLite locally) after every step, keyed by thread id.
Lives as long as the conversation does.

LONG-TERM (cross-session): what we know about the CUSTOMER across conversations —
preferences, past issues, resolutions. Two complementary stores:
- structured profile (SQLite): facts with exact lookup (plan, language, name).
- episodic memory (Qdrant collection): past interaction summaries retrieved by
  semantic similarity ("customer had a similar timeout issue in March").

WHY BOTH ARE NECESSARY
Short-term without long-term = amnesia between sessions (customer repeats
themselves — the #1 support complaint). Long-term without short-term = no
coherent conversation at all. They also differ mechanically: short-term is an
exact snapshot restored by key; long-term is a lossy, searchable distillation.
That asymmetry (snapshot vs distillation) is the crisp interview answer.

WHY WE BUILD IT BY HAND instead of using Mem0/Zep: those products are fine, but
they hide exactly the mechanism (extract -> consolidate -> retrieve) worth learning.
"""
