import json

import pytest
from pydantic import BaseModel, Field

from nimbusdesk.agents.local_tools import GetServiceStatusTool, LookupCustomerTool
from nimbusdesk.agents.tools import Tool, ToolError


class EchoInput(BaseModel):
    text: str = Field(min_length=1, max_length=10)


class EchoTool(Tool):
    name = "echo"
    description = "Echoes text back"
    input_model = EchoInput

    def execute(self, args: EchoInput) -> str:
        return args.text.upper()


def test_spec_is_generated_from_the_pydantic_model():
    spec = EchoTool().spec()
    assert spec.name == "echo"
    assert spec.input_schema["properties"]["text"]["maxLength"] == 10
    assert "text" in spec.input_schema["required"]


def test_valid_arguments_reach_execute_typed():
    assert EchoTool().run({"text": "hi"}) == "HI"


def test_invalid_arguments_raise_model_readable_tool_error():
    with pytest.raises(ToolError, match="text"):
        EchoTool().run({"text": ""})  # too short
    with pytest.raises(ToolError, match="text"):
        EchoTool().run({})  # missing entirely


def test_customer_lookup_found_and_not_found():
    tool = LookupCustomerTool()
    found = json.loads(tool.run({"email": "dana@acme.io"}))
    assert found["plan"] == "business"
    # Unknown customer is an ANSWER (relay it), not an error (retry it).
    assert "no customer found" in tool.run({"email": "ghost@nowhere.io"})


def test_customer_lookup_rejects_malformed_email():
    with pytest.raises(ToolError):
        LookupCustomerTool().run({"email": "not-an-email"})


def test_service_status_rejects_unknown_component():
    tool = GetServiceStatusTool()
    assert "degraded" in tool.run({"component": "sync"})
    with pytest.raises(ToolError):
        tool.run({"component": "database"})  # not in the Literal
