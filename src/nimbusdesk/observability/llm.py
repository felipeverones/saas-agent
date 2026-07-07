"""LLM tracing decorator — spans for every model call, same trick as cost.

Stacks with UsageTracker on the provider port (both satisfy the same
Protocols): `UsageTracker(TracingLLM(AnthropicProvider(...)))` gives token
accounting AND per-call spans with zero changes anywhere else — the promised
payoff of putting a port in front of the LLM back in phase 2.

Attribute names follow the OpenTelemetry GenAI semantic conventions
(gen_ai.*) so any OTel-aware viewer (Phoenix included) renders model,
token usage and latency without custom mapping.
"""

from typing import Sequence

from nimbusdesk.llm.ports import (
    Completion,
    Message,
    ToolSpec,
    ToolUseCompletion,
    Turn,
)
from nimbusdesk.observability.tracing import span


class TracingLLM:
    def __init__(self, inner) -> None:
        self._inner = inner

    def complete(
        self,
        *,
        messages: Sequence[Message],
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        with span("llm.complete") as current:
            completion = self._inner.complete(
                messages=messages, system=system,
                max_tokens=max_tokens, temperature=temperature,
            )
            current.set_attribute("gen_ai.request.model", completion.model)
            current.set_attribute("gen_ai.usage.input_tokens", completion.input_tokens)
            current.set_attribute("gen_ai.usage.output_tokens", completion.output_tokens)
            return completion

    def complete_with_tools(
        self,
        *,
        turns: Sequence[Turn],
        tools: Sequence[ToolSpec],
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ToolUseCompletion:
        with span("llm.complete_with_tools", tool_count=len(tools)) as current:
            completion = self._inner.complete_with_tools(
                turns=turns, tools=tools, system=system,
                max_tokens=max_tokens, temperature=temperature,
            )
            current.set_attribute("gen_ai.request.model", completion.model)
            current.set_attribute("gen_ai.usage.input_tokens", completion.input_tokens)
            current.set_attribute("gen_ai.usage.output_tokens", completion.output_tokens)
            current.set_attribute("llm.stop_reason", completion.stop_reason)
            return completion
