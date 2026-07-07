"""The agent loop: reason -> act -> observe, until done or out of budget.

THE ReAct PATTERN (Reason + Act)
An LLM alone can only talk. An agent is an LLM in a loop with tools:
1. REASON — the model reads the conversation so far and decides: answer now,
   or gather more information first?
2. ACT — if it needs information, it emits structured tool calls.
3. OBSERVE — we execute the calls and feed results back as observations.
Repeat. The loop ends when the model answers WITHOUT requesting tools
(stop_reason "end_turn") — the model itself decides when it's done, which is
what makes this an agent rather than a fixed pipeline.

THREE PRODUCTION RULES BAKED IN
- ITERATION BUDGET: an agent that can loop WILL eventually loop (tool returns
  something confusing -> model retries forever). `max_iterations` turns an
  infinite incident into a bounded, observable failure.
- ERRORS ARE OBSERVATIONS: a failed tool call goes back to the model as an
  error result (is_error=True), not up the stack as an exception. Models
  routinely recover — retry with fixed arguments, try another tool, or tell
  the user. One flaky tool must not kill the conversation.
- FULL STEP TRACE: AgentResult carries every (tool, args, observation) step.
  "Why did the agent do that?" must be answerable from data — this trace is
  what phase 8 exports as spans.
"""

import logging
from typing import Sequence

from pydantic import BaseModel, Field

from nimbusdesk.agents.tools import ToolError, ToolLike
from nimbusdesk.guardrails.injection import sanitize_observation
from nimbusdesk.llm.ports import ToolCall, ToolCallingLLM, ToolResultTurn, Turn, UserTurn
from nimbusdesk.observability.tracing import span

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 8

ITERATION_LIMIT_MESSAGE = (
    "I couldn't complete this within my step budget. "
    "Escalating to a human agent is recommended."
)


class AgentStep(BaseModel):
    """One act->observe cycle, kept for auditability."""

    tool: str
    arguments: dict
    observation: str
    is_error: bool = False


class AgentResult(BaseModel):
    answer: str
    steps: list[AgentStep] = Field(default_factory=list)
    iterations: int = 0
    hit_iteration_limit: bool = False


class ReactAgent:
    def __init__(
        self,
        llm: ToolCallingLLM,
        tools: Sequence[ToolLike],
        system_prompt: str,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        self._llm = llm
        self._tools = {tool.name: tool for tool in tools}
        self._system = system_prompt
        self._max_iterations = max_iterations

    def run(self, question: str) -> AgentResult:
        with span("agent.loop", max_iterations=self._max_iterations) as loop_span:
            result = self._run(question)
            loop_span.set_attribute("agent.iterations", result.iterations)
            loop_span.set_attribute("agent.tool_calls", len(result.steps))
            loop_span.set_attribute("agent.hit_iteration_limit", result.hit_iteration_limit)
            return result

    def _run(self, question: str) -> AgentResult:
        turns: list[Turn] = [UserTurn(content=question)]
        steps: list[AgentStep] = []
        specs = [tool.spec() for tool in self._tools.values()]

        for iteration in range(1, self._max_iterations + 1):
            completion = self._llm.complete_with_tools(
                turns=turns, tools=specs, system=self._system
            )
            turns.append(completion.turn)

            if not completion.turn.tool_calls:
                # The model answered instead of acting: the loop's exit door.
                return AgentResult(
                    answer=completion.turn.text, steps=steps, iterations=iteration
                )

            for call in completion.turn.tool_calls:
                observation, is_error = self._execute(call)
                steps.append(
                    AgentStep(
                        tool=call.name,
                        arguments=call.arguments,
                        observation=observation,
                        is_error=is_error,
                    )
                )
                turns.append(
                    ToolResultTurn(
                        tool_call_id=call.id,
                        # Indirect-injection defense (phase 7): tool results
                        # are UNTRUSTED — delimited, size-capped, and flagged
                        # if they contain instruction-like text. The raw
                        # observation stays in the step trace for audit.
                        content=sanitize_observation(observation),
                        is_error=is_error,
                    )
                )

        logger.warning("agent hit iteration limit (%d)", self._max_iterations)
        return AgentResult(
            answer=ITERATION_LIMIT_MESSAGE,
            steps=steps,
            iterations=self._max_iterations,
            hit_iteration_limit=True,
        )

    def _execute(self, call: ToolCall) -> tuple[str, bool]:
        with span(f"tool.{call.name}", tool_args=call.arguments) as tool_span:
            observation, is_error = self._execute_inner(call)
            tool_span.set_attribute("tool.is_error", is_error)
            return observation, is_error

    def _execute_inner(self, call: ToolCall) -> tuple[str, bool]:
        tool = self._tools.get(call.name)
        if tool is None:
            # Hallucinated tool name: tell the model what actually exists.
            return (
                f"unknown tool '{call.name}'; available: {', '.join(self._tools)}",
                True,
            )
        try:
            return tool.run(call.arguments), False
        except ToolError as error:
            return str(error), True
        except Exception:
            # Genuine bug/outage: log the details for US, but give the model a
            # neutral observation — stack traces confuse it and leak internals.
            logger.exception("tool '%s' crashed", call.name)
            return f"tool '{call.name}' failed unexpectedly; try another approach", True
