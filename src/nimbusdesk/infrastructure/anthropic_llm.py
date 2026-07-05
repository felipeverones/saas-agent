"""Anthropic adapter — the concrete LLMProvider behind llm/ports.py.

This is the ONLY module in the codebase allowed to import the `anthropic` SDK
(the architecture test keeps it out of domain; discipline keeps it out of the
rest). Everything above this line of the stack speaks our own Message /
Completion contract, which is what makes the test suite LLM-free.
"""

from typing import Sequence

from anthropic import Anthropic

from nimbusdesk.llm.ports import (
    AssistantTurn,
    Completion,
    Message,
    ToolCall,
    ToolSpec,
    ToolUseCompletion,
    Turn,
    UserTurn,
)


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

    def complete_with_tools(
        self,
        *,
        turns: Sequence[Turn],
        tools: Sequence[ToolSpec],
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ToolUseCompletion:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "",
            messages=_to_anthropic_messages(turns),
            tools=[
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ],
        )

        text = "".join(b.text for b in response.content if b.type == "text")
        tool_calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
            for b in response.content
            if b.type == "tool_use"
        ]
        stop = (
            response.stop_reason
            if response.stop_reason in ("tool_use", "max_tokens")
            else "end_turn"
        )
        return ToolUseCompletion(
            turn=AssistantTurn(text=text, tool_calls=tool_calls),
            stop_reason=stop,
            model=self._model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


def _to_anthropic_messages(turns: Sequence[Turn]) -> list[dict]:
    """Map our vendor-neutral turns onto Anthropic's message format.

    Wire-format detail worth knowing: tool RESULTS travel as content blocks
    inside a USER message, and consecutive results (from parallel tool calls)
    must be merged into ONE user message — the API rejects two user messages
    in a row.
    """
    messages: list[dict] = []
    for turn in turns:
        if isinstance(turn, UserTurn):
            messages.append({"role": "user", "content": turn.content})
        elif isinstance(turn, AssistantTurn):
            blocks: list[dict] = []
            if turn.text:
                blocks.append({"type": "text", "text": turn.text})
            blocks.extend(
                {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                for c in turn.tool_calls
            )
            messages.append({"role": "assistant", "content": blocks})
        else:  # ToolResultTurn
            block = {
                "type": "tool_result",
                "tool_use_id": turn.tool_call_id,
                "content": turn.content,
                "is_error": turn.is_error,
            }
            last = messages[-1] if messages else None
            if last and last["role"] == "user" and isinstance(last["content"], list):
                last["content"].append(block)
            else:
                messages.append({"role": "user", "content": [block]})
    return messages
