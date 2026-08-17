"""Tests for Agent.stream()."""

from __future__ import annotations

import httpx
import pytest

from pure_agents import Agent, StreamEvent, tool
from tests.conftest import FakeLLM, anthropic_sse, openai_sse, openai_tool_call


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


async def collect(agent: Agent, prompt: str = "hi") -> list[StreamEvent]:
    return [event async for event in agent.stream(prompt)]


class TestTextStreaming:
    async def test_text_arrives_as_typed_events(self, make_agent):
        agent, fake = make_agent()
        fake.queue_sse(openai_sse(["Hel", "lo ", "world"]))

        events = await collect(agent)

        assert [e.type for e in events] == ["text", "text", "text", "done"]
        assert "".join(e.content for e in events if e.type == "text") == "Hello world"

    async def test_done_carries_the_full_answer(self, make_agent):
        agent, fake = make_agent()
        fake.queue_sse(openai_sse(["Hel", "lo"]))

        events = await collect(agent)

        assert events[-1].type == "done"
        assert events[-1].content == "Hello"

    async def test_history_is_updated(self, make_agent):
        agent, fake = make_agent()
        fake.queue_sse(openai_sse(["Hi"]))

        await collect(agent)

        assert [m.role for m in agent.messages] == ["system", "user", "assistant"]
        assert agent.messages[-1].content == "Hi"

    async def test_anthropic_streams_too(self, make_agent):
        agent, fake = make_agent(provider="anthropic")
        fake.queue_sse(anthropic_sse(["Hola", " mundo"]))

        events = await collect(agent)

        assert "".join(e.content for e in events if e.type == "text") == "Hola mundo"
        assert events[-1].type == "done"


class TestToolEvents:
    async def test_tool_calls_and_results_are_visible(self, make_agent):
        agent, fake = make_agent(tools=[add])
        fake.queue_sse(
            openai_sse(
                ["Let me add those. "],
                [openai_tool_call("add", {"a": 2, "b": 3}, call_id="c1")],
            )
        )
        fake.queue_sse(openai_sse(["The answer is 5."]))

        events = await collect(agent, "what is 2 + 3?")

        types = [e.type for e in events]
        assert "tool_call" in types
        assert "tool_result" in types

        call = next(e for e in events if e.type == "tool_call")
        assert call.name == "add"
        assert call.arguments == {"a": 2, "b": 3}
        assert call.id == "c1"

        result = next(e for e in events if e.type == "tool_result")
        assert result.name == "add"
        assert result.content == "5"
        assert result.id == "c1"

    async def test_the_loop_continues_after_a_tool(self, make_agent):
        agent, fake = make_agent(tools=[add])
        fake.queue_sse(
            openai_sse(tool_calls=[openai_tool_call("add", {"a": 1, "b": 1})])
        )
        fake.queue_sse(openai_sse(["Two."]))

        events = await collect(agent, "add")

        assert fake.call_count == 2
        assert events[-1].content == "Two."

    async def test_anthropic_tool_streaming(self, make_agent):
        agent, fake = make_agent(provider="anthropic", tools=[add])
        fake.queue_sse(
            anthropic_sse(
                tool_uses=[{"id": "t1", "name": "add", "input": {"a": 4, "b": 5}}]
            )
        )
        fake.queue_sse(anthropic_sse(["Nine."]))

        events = await collect(agent, "4 + 5?")

        call = next(e for e in events if e.type == "tool_call")
        assert call.arguments == {"a": 4, "b": 5}
        assert next(e for e in events if e.type == "tool_result").content == "9"


class TestStreamingParity:
    async def test_tokens_are_counted(self, make_agent):
        agent, fake = make_agent()
        fake.queue_sse(openai_sse(["Hi"], prompt_tokens=12, completion_tokens=3))

        await collect(agent)

        assert agent.usage.input_tokens == 12
        assert agent.usage.output_tokens == 3
        assert agent.usage.requests == 1

    async def test_anthropic_tokens_are_counted(self, make_agent):
        agent, fake = make_agent(provider="anthropic")
        fake.queue_sse(anthropic_sse(["Hi"], input_tokens=7, output_tokens=2))

        await collect(agent)

        assert agent.usage.input_tokens == 7
        assert agent.usage.output_tokens == 2

    async def test_retries_before_the_first_chunk(self, make_agent, monkeypatch):
        monkeypatch.setattr("pure_agents.agent._retry_delay", lambda exc, attempt: 0)
        agent, fake = make_agent(retries=2)
        fake.queue_error(503)
        fake.queue_sse(openai_sse(["recovered"]))

        events = await collect(agent)

        assert events[-1].content == "recovered"
        assert fake.call_count == 2

    async def test_a_401_is_not_retried(self, make_agent):
        agent, fake = make_agent(retries=3)
        fake.queue_error(401)

        with pytest.raises(httpx.HTTPStatusError):
            await collect(agent)

        assert fake.call_count == 1

    async def test_it_falls_back_to_the_other_provider(self):
        agent = Agent(
            api_key="k",
            provider="mistral",
            fallback="anthropic",
            fallback_api_key="a",
        )
        primary, secondary = FakeLLM(), FakeLLM()
        agent.client.transport = primary.transport()
        agent.fallback_client.transport = secondary.transport()

        primary.queue_error(500)
        secondary.queue_sse(anthropic_sse(["rescued"]))

        events = await collect(agent)

        assert events[-1].content == "rescued"
        assert secondary.last_request()["model"] == "claude-sonnet-5"

    async def test_max_steps_ends_the_stream(self, make_agent):
        agent, fake = make_agent(tools=[add], max_steps=2)
        for _ in range(2):
            fake.queue_sse(
                openai_sse(tool_calls=[openai_tool_call("add", {"a": 1, "b": 1})])
            )

        events = await collect(agent, "loop")

        assert events[-1].type == "done"
        assert "Max steps" in events[-1].content

    async def test_the_thinking_hook_fires(self, make_agent):
        seen: list[str] = []
        agent, fake = make_agent(on_thinking=seen.append)
        fake.queue_sse(openai_sse(["Thought"]))

        await collect(agent)

        assert seen == ["Thought"]
