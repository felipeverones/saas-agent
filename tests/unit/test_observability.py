"""Tracing and cost accounting.

The tracing test uses OTel's InMemorySpanExporter: a real TracerProvider,
real spans, exported to a list instead of the network — so we assert the
actual span tree a run produces, not mocks of it.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel, Field

from nimbusdesk.agents.react import ReactAgent
from nimbusdesk.agents.tools import Tool
from nimbusdesk.llm.ports import AssistantTurn, ToolCall
from nimbusdesk.observability.cost import estimate_usd, format_usage
from nimbusdesk.observability.llm import TracingLLM
from tests.fakes import FakeToolLLM

# One provider per test session: OTel's global provider is set-once.
_EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
trace.set_tracer_provider(_provider)


class PingInput(BaseModel):
    target: str = Field(min_length=1)


class PingTool(Tool):
    name = "ping"
    description = "ping something"
    input_model = PingInput

    def execute(self, args: PingInput) -> str:
        return f"{args.target}: ok"


def test_agent_run_emits_the_span_tree():
    _EXPORTER.clear()
    llm = TracingLLM(
        FakeToolLLM(
            [
                AssistantTurn(
                    tool_calls=[ToolCall(id="c1", name="ping", arguments={"target": "api"})]
                ),
                AssistantTurn(text="api is up"),
            ]
        )
    )
    ReactAgent(llm, [PingTool()], "sys").run("is the api up?")

    spans = {s.name: s for s in _EXPORTER.get_finished_spans()}
    assert "agent.loop" in spans
    assert "tool.ping" in spans
    assert "llm.complete_with_tools" in spans

    assert spans["agent.loop"].attributes["agent.iterations"] == 2
    assert spans["tool.ping"].attributes["tool.is_error"] is False
    assert spans["llm.complete_with_tools"].attributes["gen_ai.usage.input_tokens"] == 10
    # Causality: the tool span must be a child inside the agent loop's trace.
    assert spans["tool.ping"].context.trace_id == spans["agent.loop"].context.trace_id


def test_cost_estimation_and_formatting():
    # 1M input tokens on the fast tier = $1.00 by the table.
    assert estimate_usd("claude-haiku-4-5-20251001", 1_000_000, 0) == 1.0
    # Unknown models fall back to strong-tier pricing (overestimate on purpose).
    assert estimate_usd("mystery-model", 1_000_000, 0) == 3.0

    line = format_usage([("claude-sonnet-5", 1000, 500), ("claude-haiku-4-5-20251001", 2000, 100)])
    assert "3000 in / 600 out" in line
    assert "$" in line
