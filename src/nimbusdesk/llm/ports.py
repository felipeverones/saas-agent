"""The LLM port: what the application needs from a language model — no more.

Kept deliberately minimal (messages in, text + token counts out). Structured
tool-calling arrives with the agent loop in phase 3; adding capabilities to a
port before a consumer exists is how interfaces rot.
"""

from typing import Literal, Protocol, Sequence

from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


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
