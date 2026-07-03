"""Usage tracking — the decorator pattern applied to the LLM port.

THE PROBLEM: a single answer involves several LLM calls (rewrite, generate,
self-check), each made by a different component. Threading token counts
through every return type pollutes every API with an accounting concern.

THE FIX: wrap the provider once. UsageTracker satisfies the same LLMProvider
Protocol it wraps (structural typing again), so components use it unknowingly
while it accumulates totals on the side. One wrapper at the composition root
instead of bookkeeping in N classes — this is also exactly where phase 8
attaches tracing spans and per-model cost tables.
"""

from typing import Sequence

from nimbusdesk.llm.ports import Completion, LLMProvider, Message


class UsageTracker:
    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def complete(
        self,
        *,
        messages: Sequence[Message],
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        completion = self._inner.complete(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.input_tokens += completion.input_tokens
        self.output_tokens += completion.output_tokens
        self.calls += 1
        return completion
