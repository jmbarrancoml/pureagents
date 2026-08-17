"""Tests for the core Agent.run() loop against scripted provider responses."""

from __future__ import annotations

import pytest

from pure_agents import MaxStepsError, tool
from tests.conftest import (
    anthropic_response,
    anthropic_tool_use,
    openai_response,
    openai_tool_call,
)


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def shout(text: str) -> str:
    """Uppercase some text."""
    return text.upper()


class TestBasicRun:
    async def test_returns_final_content(self, make_agent):
        agent, fake = make_agent()
        fake.queue(openai_response("Madrid is sunny."))

        result = await agent.run("Weather in Madrid?")

        assert result == "Madrid is sunny."
        assert fake.call_count == 1

    async def test_sends_system_prompt_first(self, make_agent):
        agent, fake = make_agent(system_prompt="You are a pirate.")
        fake.queue(openai_response("Arr."))

        await agent.run("Hello")

        messages = fake.last_request()["messages"]
        assert messages[0] == {"role": "system", "content": "You are a pirate."}
        assert messages[1] == {"role": "user", "content": "Hello"}

    async def test_history_persists_across_calls(self, make_agent):
        agent, fake = make_agent()
        fake.queue(openai_response("Hi Jose."))
        fake.queue(openai_response("Your name is Jose."))

        await agent.run("I'm Jose")
        await agent.run("What's my name?")

        roles = [m["role"] for m in fake.last_request()["messages"]]
        assert roles == ["system", "user", "assistant", "user"]

    async def test_usage_accumulates_across_requests(self, make_agent):
        agent, fake = make_agent()
        fake.queue(openai_response("one", prompt_tokens=10, completion_tokens=3))
        fake.queue(openai_response("two", prompt_tokens=20, completion_tokens=7))

        await agent.run("a")
        await agent.run("b")

        assert agent.usage.input_tokens == 30
        assert agent.usage.output_tokens == 10
        assert agent.usage.total_tokens == 40
        assert agent.usage.requests == 2


class TestToolLoop:
    async def test_executes_tool_then_answers(self, make_agent):
        agent, fake = make_agent(tools=[add])
        fake.queue(openai_response(None, [openai_tool_call("add", {"a": 2, "b": 3})]))
        fake.queue(openai_response("The answer is 5."))

        result = await agent.run("What is 2 + 3?")

        assert result == "The answer is 5."
        assert fake.call_count == 2

        follow_up = fake.last_request()["messages"]
        tool_message = follow_up[-1]
        assert tool_message["role"] == "tool"
        assert tool_message["content"] == "5"
        assert tool_message["tool_call_id"] == "call_1"

    async def test_advertises_tools_in_the_request(self, make_agent):
        agent, fake = make_agent(tools=[add])
        fake.queue(openai_response("done"))

        await agent.run("hi")

        payload = fake.last_request()
        assert payload["tool_choice"] == "auto"
        assert [t["function"]["name"] for t in payload["tools"]] == ["add"]

    async def test_runs_parallel_tool_calls(self, make_agent):
        agent, fake = make_agent(tools=[add, shout])
        fake.queue(
            openai_response(
                None,
                [
                    openai_tool_call("add", {"a": 1, "b": 1}, call_id="c1"),
                    openai_tool_call("shout", {"text": "hey"}, call_id="c2"),
                ],
            )
        )
        fake.queue(openai_response("both done"))

        result = await agent.run("do both")

        assert result == "both done"
        tool_messages = [
            m for m in fake.last_request()["messages"] if m["role"] == "tool"
        ]
        assert [m["tool_call_id"] for m in tool_messages] == ["c1", "c2"]
        assert [m["content"] for m in tool_messages] == ["2", "HEY"]

    async def test_unknown_tool_is_reported_back_to_the_model(self, make_agent):
        agent, fake = make_agent(tools=[add])
        fake.queue(openai_response(None, [openai_tool_call("nope", {})]))
        fake.queue(openai_response("recovered"))

        result = await agent.run("call something odd")

        assert result == "recovered"
        tool_message = fake.last_request()["messages"][-1]
        assert "Unknown tool" in tool_message["content"]

    async def test_tool_exception_is_reported_back_to_the_model(self, make_agent):
        @tool
        def explode(x: str) -> str:
            """Always fails."""
            raise RuntimeError("kaboom")

        agent, fake = make_agent(tools=[explode])
        fake.queue(openai_response(None, [openai_tool_call("explode", {"x": "a"})]))
        fake.queue(openai_response("handled"))

        assert await agent.run("go") == "handled"
        assert "kaboom" in fake.last_request()["messages"][-1]["content"]

    async def test_stops_at_max_steps(self, make_agent):
        agent, fake = make_agent(tools=[add], max_steps=2)
        for _ in range(2):
            fake.queue(
                openai_response(None, [openai_tool_call("add", {"a": 1, "b": 1})])
            )

        with pytest.raises(MaxStepsError) as excinfo:
            await agent.run("loop forever")

        assert fake.call_count == 2
        assert excinfo.value.steps == 2


class TestAnthropicProvider:
    async def test_returns_final_content(self, make_agent):
        agent, fake = make_agent(provider="anthropic")
        fake.queue(anthropic_response("Hello from Anthropic."))

        assert await agent.run("hi") == "Hello from Anthropic."

    async def test_system_prompt_is_a_top_level_field(self, make_agent):
        agent, fake = make_agent(provider="anthropic", system_prompt="Be terse.")
        fake.queue(anthropic_response("ok"))

        await agent.run("hi")

        payload = fake.last_request()
        assert payload["system"] == "Be terse."
        assert payload["messages"] == [{"role": "user", "content": "hi"}]

    async def test_executes_tool_then_answers(self, make_agent):
        agent, fake = make_agent(provider="anthropic", tools=[add])
        fake.queue(
            anthropic_response(
                tool_uses=[anthropic_tool_use("add", {"a": 4, "b": 5})],
                stop_reason="tool_use",
            )
        )
        fake.queue(anthropic_response("It is 9."))

        assert await agent.run("4 + 5?") == "It is 9."

        tool_result = fake.last_request()["messages"][-1]
        assert tool_result["role"] == "user"
        assert tool_result["content"][0]["type"] == "tool_result"
        assert tool_result["content"][0]["content"] == "9"

    async def test_usage_is_tracked(self, make_agent):
        agent, fake = make_agent(provider="anthropic")
        fake.queue(anthropic_response("ok", input_tokens=11, output_tokens=4))

        await agent.run("hi")

        assert agent.usage.input_tokens == 11
        assert agent.usage.output_tokens == 4


class TestHarness:
    async def test_unscripted_request_fails_loudly(self, make_agent):
        agent, _ = make_agent()
        with pytest.raises(Exception):
            await agent.run("nothing queued")
