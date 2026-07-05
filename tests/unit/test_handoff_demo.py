from nimbusdesk.agents.handoff_demo import HandoffAgentSpec, HandoffRunner
from nimbusdesk.llm.ports import AssistantTurn, ToolCall
from tests.fakes import FakeToolLLM

FRONTDESK = HandoffAgentSpec(
    name="frontdesk",
    system_prompt="You greet and route.",
    handoff_targets=["billing"],
)
BILLING = HandoffAgentSpec(name="billing", system_prompt="You handle billing.")


def test_transfer_switches_agent_and_records_trail():
    llm = FakeToolLLM(
        [
            AssistantTurn(tool_calls=[ToolCall(id="c1", name="transfer_to_billing", arguments={})]),
            AssistantTurn(text="Refunds take 30 days."),
        ]
    )
    result = HandoffRunner(llm, [FRONTDESK, BILLING], entry="frontdesk").run("refund?")

    assert result.agent_trail == ["frontdesk", "billing"]
    assert result.answer == "Refunds take 30 days."
    # After the transfer, the model must be running under the TARGET's prompt —
    # the persona swap is the whole point of the pattern.
    assert llm.calls[1]["system"] == "You handle billing."
    # And the frontdesk agent must have been OFFERED the handoff tool.
    assert any(t.name == "transfer_to_billing" for t in llm.calls[0]["tools"])


def test_turn_budget_stops_transfer_ping_pong():
    # Two agents forwarding to each other forever — the known failure mode of
    # direct handoffs, which only a budget can stop.
    a = HandoffAgentSpec(name="a", system_prompt="a", handoff_targets=["b"])
    b = HandoffAgentSpec(name="b", system_prompt="b", handoff_targets=["a"])
    ping_pong = [
        AssistantTurn(
            tool_calls=[
                ToolCall(
                    id=f"c{i}",
                    name=f"transfer_to_{'b' if i % 2 == 0 else 'a'}",
                    arguments={},
                )
            ]
        )
        for i in range(20)
    ]
    result = HandoffRunner(FakeToolLLM(ping_pong), [a, b], entry="a").run("hi")

    assert result.hit_turn_limit
    assert len(result.agent_trail) > 2