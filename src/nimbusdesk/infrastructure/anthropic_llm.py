"""Anthropic adapter — the concrete LLMProvider behind llm/ports.py.

This is the ONLY module in the codebase allowed to import the `anthropic` SDK
(the architecture test keeps it out of domain; discipline keeps it out of the
rest). Everything above this line of the stack speaks our own Message /
Completion contract, which is what makes the test suite LLM-free.
"""

from typing import Sequence

from anthropic import Anthropic

from nimbusdesk.llm.ports import Completion, Message


class MissingApiKeyError(RuntimeError):
    """Raised at construction time — fail at startup, not mid-request."""


class AnthropicProvider:
    def __init__(self, api_key: str | None, model: str) -> None:
        if not api_key:
            raise MissingApiKeyError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
                "your key (tests never need it — they use FakeLLMProvider)."
            )
        self._client = Anthropic(api_key=api_key)
        self._model = model

    def complete(
        self,
        *,
        messages: Sequence[Message],
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "",
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return Completion(
            text=text,
            model=self._model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
