"""The LLM port: what the application needs from a language model — no more.

Two capability levels, two protocols:
- LLMProvider: plain text completion (all the RAG pipeline needs).
- ToolCallingLLM: completion that may request TOOL CALLS (what agents need).

WHY TOOL CALLING IS A FIRST-CLASS API (not "parse the model's text")
Early agents (2023) asked the model to emit "Action: search[query]" as text
and regex-parsed it — brittle and injectable. Modern providers train models to
emit STRUCTURED tool-call blocks against a declared JSON Schema, and return
them as data, not prose. We model that contract here, vendor-neutrally.
"""

from typing import Any, Literal, Protocol, Sequence

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ToolSpec(BaseModel):
    """What the model is told about one tool: name, when to use it, and the
    JSON Schema of its arguments (generated from a Pydantic model)."""

    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(BaseModel):
    """The model's structured request to run one tool. `id` links the eventual
    result back to this exact call (models may request several in one turn)."""

    id: str
    name: str
    arguments: dict[str, Any]


class UserTurn(BaseModel):
    kind: Literal["user"] = "user"
    content: str


class AssistantTurn(BaseModel):
    kind: Literal["assistant"] = "assistant"
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolResultTurn(BaseModel):
    """The observation fed back to the model after executing a tool call.
    `is_error=True` is a legitimate observation — agents are expected to read
    errors and adapt, not crash (see agents/react.py)."""

    kind: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    content: str
    is_error: bool = False


Turn = UserTurn | AssistantTurn | ToolResultTurn


class ToolUseCompletion(BaseModel):
    turn: AssistantTurn
    stop_reason: Literal["tool_use", "end_turn", "max_tokens"]
    model: str
    input_tokens: int
    output_tokens: int


class Completion(BaseModel):
    text: str
    model: str
    # Token counts are returned by every provider and are the raw material of
    # cost tracking — capturing them from day one costs nothing; retrofitting
    # them across a codebase later is a chore nobody schedules.
    input_tokens: int
    output_tokens: int


class LLMProvider(Protocol):
    def complete(
        self,
        *,
        messages: Sequence[Message],
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion: ...


class ToolCallingLLM(Protocol):
    def complete_with_tools(
        self,
        *,
        turns: Sequence[Turn],
        tools: Sequence[ToolSpec],
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ToolUseCompletion: ...
