"""Tests for the Anthropic wire-format conversion."""

from __future__ import annotations

import pytest

from pure_agents import Agent, Message, tool
from pure_agents.clients import AnthropicClient, TruncatedResponseError
from tests.conftest import anthropic_response, anthropic_tool_use, openai_response


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def convert(messages: list[Message]):
    return AnthropicClient(api_key="k")._convert_messages(messages)[1]


class TestToolResultGrouping:
    def test_parallel_results_share_one_user_message(self):
        converted = convert(
            [
                Message(role="user", content="do both"),
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": "t1",
                            "function": {"name": "add", "arguments": "{}"},
                        },
                        {
                            "id": "t2",
                            "function": {"name": "add", "arguments": "{}"},
                        },
                    ],
                ),
                Message(role="tool", content="1", tool_call_id="t1", name="add"),
                Message(role="tool", content="2", tool_call_id="t2", name="add"),
            ]
        )

        assert [m["role"] for m in converted] == ["user", "assistant", "user"]
        results = converted[-1]["content"]
        assert [r["tool_use_id"] for r in results] == ["t1", "t2"]

    def test_roles_always_alternate(self):
        converted = convert(
            [
                Message(role="system", content="sys"),
                Message(role="user", content="q"),
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {"id": f"t{i}", "function": {"name": "add", "arguments": "{}"}}
                        for i in range(3)
                    ],
                ),
                *[
                    Message(
                        role="tool", content=str(i), tool_call_id=f"t{i}", name="add"
                    )
                    for i in range(3)
                ],
                Message(role="assistant", content="done"),
            ]
        )

        roles = [m["role"] for m in converted]
        assert all(a != b for a, b in zip(roles, roles[1:]))

    def test_a_user_turn_after_results_starts_a_new_message(self):
        converted = convert(
            [
                Message(role="user", content="q"),
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {"id": "t1", "function": {"name": "add", "arguments": "{}"}}
                    ],
                ),
                Message(role="tool", content="1", tool_call_id="t1", name="add"),
                Message(role="user", content="thanks"),
            ]
        )

        assert converted[-1] == {"role": "user", "content": "thanks"}
        assert len(converted[-2]["content"]) == 1

    async def test_parallel_tool_calls_end_to_end(self, make_agent):
        agent, fake = make_agent(provider="anthropic", tools=[add])
        fake.queue(
            anthropic_response(
                tool_uses=[
                    anthropic_tool_use("add", {"a": 1, "b": 1}, use_id="t1"),
                    anthropic_tool_use("add", {"a": 2, "b": 2}, use_id="t2"),
                ],
                stop_reason="tool_use",
            )
        )
        fake.queue(anthropic_response("both done"))

        assert await agent.run("do both") == "both done"

        messages = fake.last_request()["messages"]
        roles = [m["role"] for m in messages]
        assert all(a != b for a, b in zip(roles, roles[1:]))
        assert [b["content"] for b in messages[-1]["content"]] == ["2", "4"]


class TestMaxTokens:
    def test_defaults_to_4096(self):
        assert AnthropicClient(api_key="k").max_tokens == 4096

    async def test_agent_can_raise_it(self, make_agent):
        agent, fake = make_agent(provider="anthropic", max_tokens=32000)
        fake.queue(anthropic_response("ok"))

        await agent.run("hi")

        assert fake.last_request()["max_tokens"] == 32000

    async def test_openai_only_sends_it_when_asked(self, make_agent):
        agent, fake = make_agent()
        fake.queue(openai_response("ok"))
        fake.queue(openai_response("ok"))

        await agent.run("hi")
        assert "max_tokens" not in fake.last_request()

        capped = Agent(api_key="k", provider="openai", max_tokens=500)
        capped.client.transport = fake.transport()
        await capped.run("hi")
        assert fake.last_request()["max_tokens"] == 500


class TestTruncation:
    async def test_truncated_tool_call_raises(self, make_agent):
        agent, fake = make_agent(provider="anthropic", tools=[add])
        fake.queue(
            anthropic_response(
                tool_uses=[anthropic_tool_use("add", {"a": 1, "b": 2})],
                stop_reason="max_tokens",
            )
        )

        with pytest.raises(TruncatedResponseError, match="max_tokens"):
            await agent.run("hi")

    async def test_truncated_text_is_flagged_in_the_reply(self, make_agent):
        agent, fake = make_agent(provider="anthropic")
        fake.queue(anthropic_response("half an ans", stop_reason="max_tokens"))

        result = await agent.run("hi")

        assert result.startswith("half an ans")
        assert "truncated" in result
