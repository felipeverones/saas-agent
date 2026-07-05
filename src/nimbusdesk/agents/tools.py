"""Tool abstraction — how an agent's capabilities are declared and validated.

WHAT A TOOL IS
A function the LLM can request to run, described by (name, when-to-use-it
description, JSON Schema of arguments). The model never executes anything —
it emits a structured request; OUR code validates and executes. That split is
the security boundary of every agent system.

WHY EVERY TOOL HAS A PYDANTIC INPUT MODEL
1. The JSON Schema shown to the model is GENERATED from the model
   (`model_json_schema()`) — docs and validation can't drift apart.
2. Arguments are validated BEFORE execution. LLMs produce malformed or
   out-of-range arguments routinely — and argument injection (a manipulated
   model emitting hostile arguments) is a real attack surface. Schema
   validation is the first guardrail (phase 7 adds more).
3. Validation failures are returned to the MODEL as error observations, not
   raised to the user: a good agent reads the error and corrects itself.
"""

from pydantic import BaseModel, ValidationError

from nimbusdesk.llm.ports import ToolSpec


class ToolError(Exception):
    """A tool-level failure whose message is SAFE to show to the model.

    Distinct from arbitrary exceptions (bugs, outages) so the agent loop can
    decide what the model gets to see — internal stack traces leak
    implementation details and confuse the model; ToolError messages guide it.
    """


class Tool:
    """Base class: subclasses set the class attributes and implement execute().

    Subclasses receive an already-VALIDATED instance of `input_model` — by the
    time execute() runs, arguments are well-typed or the call never happened.
    """

    name: str
    description: str
    input_model: type[BaseModel]

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_model.model_json_schema(),
        )

    def run(self, arguments: dict) -> str:
        try:
            validated = self.input_model.model_validate(arguments)
        except ValidationError as error:
            # Compact, model-readable error — enough to self-correct on the
            # next iteration without dumping pydantic's full report.
            issues = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()
            )
            raise ToolError(f"invalid arguments — {issues}") from None
        return self.execute(validated)

    def execute(self, args: BaseModel) -> str:  # pragma: no cover - abstract
        raise NotImplementedError
