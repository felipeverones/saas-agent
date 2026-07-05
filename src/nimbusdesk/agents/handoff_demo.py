"""The ALTERNATIVE orchestration pattern: direct agent-to-agent handoff.

THE PATTERN (popularized by OpenAI's Swarm/Agents SDK)
There is no supervisor. Each agent gets, besides its real tools, special
`transfer_to_<other>` tools. When the model calls one, the runner swaps the
active agent (new system prompt + tool set) and the CONVERSATION CONTINUES —
same message history, new persona. Routing is a decision the agents make
mid-flight, not a policy at a hub.

HANDOFF vs SUPERVISOR GRAPH — the honest comparison (interview gold):
- Handoff is LESS code and feels natural: "billing question? pass it to Bob."
- But control flow is IMPLICIT: who can reach whom is buried in tool lists,
  there's no central place to audit or cap routing, and two agents can ping-
  pong a ticket forever unless you add budgets (we do).
- The graph makes flow EXPLICIT and checkpointable at the cost of ceremony.
Rule of thumb: handoffs for small conversational systems (2-3 personas),
state graph when you need auditability, checkpoints, or human-in-the-loop.

This module is deliberately minimal — it exists so both models can be FELT,
not to be the production path (that's graph.py).
"""

from pydantic import BaseModel, Field

from nimbusdesk.agents.tools import Tool, ToolError
from nimbusdesk.llm.ports import ToolCallingLLM, ToolResultTurn, Turn, UserTurn

MAX_TURNS = 10
HANDOFF_PREFIX = "transfer_to_"


class HandoffAgentSpec(BaseModel):
    name: str
    system_prompt: str
    handoff_targets: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}
    tools: list[Tool] = Field(default_factory=list)


class HandoffResult(BaseModel):
    answer: str
    # The audit trail: which personas touched the conversation, in order.
    agent_trail: list[str]
    hit_turn_limit: bool = False


class HandoffRunner:
    def __init__(self, llm: ToolCallingLLM, agents: list[HandoffAgentSpec], entry: str) -> None:
        self._llm = llm
        self._agents = {a.name: a for a in agents}
        self._entry = entry

    def run(self, question: str) -> HandoffResult:
        active = self._agents[self._entry]
        trail = [active.name]
        turns: list[Turn] = [UserTurn(content=question)]

        for _ in range(MAX_TURNS):
            specs = [t.spec() for t in active.tools] + [
                _handoff_spec(target) for target in active.handoff_targets
            ]
            completion = self._llm.complete_with_tools(
                turns=turns, tools=specs, system=active.system_prompt
            )
            turns.append(completion.turn)

            if not completion.turn.tool_calls:
                return HandoffResult(answer=completion.turn.text, agent_trail=trail)

            for call in completion.turn.tool_calls:
                if call.name.startswith(HANDOFF_PREFIX):
                    target = call.name.removeprefix(HANDOFF_PREFIX)
                    # THE handoff moment: swap persona, keep the conversation.
                    active = self._agents[target]
                    trail.append(target)
                    observation = f"conversation transferred to {target}"
                    turns.append(
                        ToolResultTurn(tool_call_id=call.id, content=observation)
                    )
                    continue
                tool = next((t for t in active.tools if t.name == call.name), None)
                if tool is None:
                    turns.append(
                        ToolResultTurn(
                            tool_call_id=call.id,
                            content=f"unknown tool '{call.name}'",
                            is_error=True,
                        )
                    )
                    continue
                try:
                    turns.append(
                        ToolResultTurn(tool_call_id=call.id, content=tool.run(call.arguments))
                    )
                except ToolError as error:
                    turns.append(
                        ToolResultTurn(
                            tool_call_id=call.id, content=str(error), is_error=True
                        )
                    )

        return HandoffResult(
            answer="Conversation exceeded the turn budget.",
            agent_trail=trail,
            hit_turn_limit=True,
        )


def _handoff_spec(target: str):
    from nimbusdesk.llm.ports import ToolSpec

    return ToolSpec(
        name=f"{HANDOFF_PREFIX}{target}",
        description=f"Transfer this conversation to the {target} agent when the "
        f"request falls under their specialty.",
        input_schema={"type": "object", "properties": {}, "required": []},
    )
