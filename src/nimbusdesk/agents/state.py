"""The shared, typed state of the multi-agent graph.

WHY TYPED STATE (Pydantic) AND NOT LOOSE STRINGS/DICTS
In a multi-agent system, the state IS the interface between agents: the triage
agent writes `triage`, the supervisor reads it, specialists write
`final_answer`. With a schema, an agent writing garbage fails loudly at the
boundary; with free-form dicts, garbage propagates silently and surfaces three
agents later as a weird answer — the single worst debugging experience in
multi-agent systems.

HOW LANGGRAPH USES THIS
Each node receives the CURRENT state and returns a PARTIAL update (a dict of
just the fields it changed); LangGraph validates and merges the update, then
CHECKPOINTS the result before the next node runs. That per-step persistence is
what enables pause/resume (phase 7's human approval) and crash recovery.
"""

import operator
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from nimbusdesk.domain.support import TriageDecision


class ChatTurn(BaseModel):
    """One utterance in the running conversation (short-term memory unit)."""

    role: Literal["customer", "assistant"]
    content: str


class SupportState(BaseModel):
    # -- the request ---------------------------------------------------------
    question: str
    customer_email: str | None = None

    # -- short-term memory (phase 6) ------------------------------------------
    # Annotated with a REDUCER: nodes return just the NEW turns and LangGraph
    # APPENDS them (operator.add) instead of replacing — this is how the
    # conversation accumulates across graph invocations on the same thread,
    # while per-turn fields below are simply overwritten each turn.
    history: Annotated[list[ChatTurn], operator.add] = Field(default_factory=list)
    turn_index: int = 0

    # -- long-term memory context (phase 6) ------------------------------------
    # Written by the recall node at the start of each turn: profile facts +
    # relevant past episodes for THIS customer, or None when nothing is known.
    memory_context: str | None = None
    # The conversation's thread id, mirrored into state so the finalize node
    # can key episodic memories. (Nodes can't see the invoke config directly.)
    thread_hint: str | None = None

    # -- written by triage ---------------------------------------------------
    triage: TriageDecision | None = None

    # -- written by specialists ----------------------------------------------
    final_answer: str | None = None
    resolved_by: str | None = None

    # -- escalation ----------------------------------------------------------
    escalated: bool = False
    escalation_reason: str | None = None

    # -- control & audit (written by the supervisor / failure handling) -------
    # Failures are DATA, not exceptions: a specialist crashing is recorded
    # here and routed around (to escalation), never allowed to kill the run.
    failures: list[str] = Field(default_factory=list)
    # Supervisor visit counter — the graph-level iteration budget, same
    # principle as ReactAgent.max_iterations one level down.
    supervisor_visits: int = 0
