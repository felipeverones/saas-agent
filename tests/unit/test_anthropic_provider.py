"""Adapter tests against a MOCKED Anthropic SDK.

We can't (and shouldn't) hit the real API in tests — but the adapter still has
logic that can break: request field mapping, response block parsing, usage
extraction. Mocking the SDK client pins all of it, so when a real key shows
up, the only untested surface is Anthropic's own server.
"""

from types import SimpleNamespace

import pytest

from nimbusdesk.infrastructure.anthropic_llm import AnthropicProvider, MissingApiKeyError
from nimbusdesk.llm.ports import Message


def _fake_response(*blocks, input_tokens=7, output_tokens=3):
    return SimpleNamespace(
        content=list(blocks),
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason="end_turn",
    )


def test_missing_key_fails_at_construction_with_remediation():
    with pytest.raises(MissingApiKeyError, match=r"\.env"):
        AnthropicProvider(api_key=None, model="claude-sonnet-5")
    with pytest.raises(MissingApiKeyError):
        AnthropicProvider(api_key="", model="claude-sonnet-5")


def test_complete_maps_request_and_response(mocker):
    client_cls = mocker.patch("nimbusdesk.infrastructure.anthropic_llm.Anthropic")
    client = client_cls.return_value
    client.messages.create.return_value = _fake_response(
        SimpleNamespace(type="text", text="Hello "),
        SimpleNamespace(type="text", text="world"),
    )

    provider = AnthropicProvider(api_key="sk-test", model="claude-sonnet-5")
    completion = provider.complete(
        messages=[Message(role="user", content="hi")], system="be brief", max_tokens=99
    )

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-5"
    assert kwargs["system"] == "be brief"
    assert kwargs["max_tokens"] == 99
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]

    assert completion.text == "Hello world"
    assert completion.input_tokens == 7
    assert completion.output_tokens == 3


def test_non_text_blocks_are_skipped(mocker):
    client_cls = mocker.patch("nimbusdesk.infrastructure.anthropic_llm.Anthropic")
    client_cls.return_value.messages.create.return_value = _fake_response(
        SimpleNamespace(type="thinking", thinking="…"),
        SimpleNamespace(type="text", text="answer"),
    )

    provider = AnthropicProvider(api_key="sk-test", model="m")
    assert provider.complete(messages=[Message(role="user", content="q")]).text == "answer"
