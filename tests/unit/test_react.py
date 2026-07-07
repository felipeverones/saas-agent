"""The agent loop's contract, pinned: exit conditions, error-as-observation,
iteration budget, and the conversation state the model actually sees."""

from pydantic import BaseModel, Field

from nimbusdesk.agents.react import ITERATION_LIMIT_MESSAGE, ReactAgent
from nimbusdesk.agents.tools import Tool
from nimbusdesk.llm.ports import AssistantTurn, ToolCall, ToolResultTurn
from tests.fakes import FakeToolLLM


class StatusInput(BaseModel):
    component: str = Field(min_length=2)


class StatusTool(Tool):
    name = "get_status"
    description = "Check a component status"
    input_model = StatusInput

    def __init__(self) -> None:
        self.invocations: list[str] = []

    def execute(self, args: StatusInput) -> str:
        self.invocations.append(args.component)
        return f"{args.component}: degraded"


def _call(name: str, arguments: dict, call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def test_direct_answer_without_tools():
    llm = FakeToolLLM([AssistantTurn(text="Refunds take 30 days.")])
    result = ReactAgent(llm, [StatusTool()], "be helpful").run("refund window?")

    assert result.answer == "Refunds take 30 days."
    assert result.iterations == 1
    assert result.steps == [] and not result.hit_iteration_limit


def test_tool_call_then_answer_feeds_observation_back():
    tool = StatusTool()
    llm = FakeToolLLM(
        [
            AssistantTurn(tool_calls=[_call("get_status", {"component": "sync"})]),
            AssistantTurn(text="Sync is degraded right now."),
        ]
    )
    result = ReactAgent(llm, [tool], "be helpful").run("is sync down?")

    assert tool.invocations == ["sync"]
    assert result.iterations == 2
    assert result.steps[0].observation == "sync: degraded"
    # The observation must reach the model on the second call, linked by id —
    # wrapped in the phase-7 untrusted-content delimiters.
    second_call_turns = llm.calls[1]["turns"]
    tool_results = [t for t in second_call_turns if isinstance(t, ToolResultTurn)]
    assert tool_results[0].tool_call_id == "c1"
    assert "sync: degraded" in tool_results[0].content
    assert tool_results[0].content.startswith("<tool_output>")


def test_invalid_arguments_become_error_observation_not_crash():
    llm = FakeToolLLM(
        [
            AssistantTurn(tool_calls=[_call("get_status", {"component": "x"})]),  # too short
            AssistantTurn(text="Sorry, I hit an issue."),
        ]
    )
    result = ReactAgent(llm, [StatusTool()], "be helpful").run("status?")

    assert result.steps[0].is_error
    assert "invalid arguments" in result.steps[0].observation
    assert result.answer == "Sorry, I hit an issue."  # loop survived


def test_hallucinated_tool_name_lists_available_tools():
    llm = FakeToolLLM(
        [
            AssistantTurn(tool_calls=[_call("reboot_server", {})]),
            AssistantTurn(text="ok"),
        ]
    )
    result = ReactAgent(llm, [StatusTool()], "be helpful").run("fix it")

    assert result.steps[0].is_error
    assert "get_status" in result.steps[0].observation


def test_iteration_budget_bounds_a_looping_agent():
    # A model that ALWAYS asks for tools would loop forever without a budget.
    endless = [
        AssistantTurn(tool_calls=[_call("get_status", {"component": "sync"}, f"c{i}")])
        for i in range(10)
    ]
    result = ReactAgent(FakeToolLLM(endless), [StatusTool()], "x", max_iterations=3).run("q")

    assert result.hit_iteration_limit
    assert result.iterations == 3
    assert result.answer == ITERATION_LIMIT_MESSAGE


def test_parallel_tool_calls_all_execute():
    tool = StatusTool()
    llm = FakeToolLLM(
        [
            AssistantTurn(
                tool_calls=[
                    _call("get_status", {"component": "sync"}, "c1"),
                    _call("get_status", {"component": "api"}, "c2"),
                ]
            ),
            AssistantTurn(text="done"),
        ]
    )
    result = ReactAgent(llm, [tool], "x").run("check both")

    assert tool.invocations == ["sync", "api"]
    assert [s.tool for s in result.steps] == ["get_status", "get_status"]
