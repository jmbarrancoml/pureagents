"""Tests for batch(), max-step signalling and session storage."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from pure_agents import Agent, JSONMemory, MaxStepsError, Message, tool
from pure_agents.memory import write_json_atomically
from tests.conftest import openai_response, openai_tool_call


@tool(group="web")
def search(query: str) -> str:
    """Search the web."""
    return f"results for {query}"


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class TestBatchCloning:
    def test_the_clone_keeps_the_full_configuration(self):
        seen: list[str] = []
        agent = Agent(
            api_key="k",
            provider="openai",
            tools=[search],
            enabled_groups=["web"],
            max_messages=8,
            max_tokens=1000,
            validator=lambda text: True,
            validation_retries=2,
            on_tool_call=lambda name, args: seen.append(name),
        )

        clone = agent._clone()

        assert clone.enabled_groups == ["web"]
        assert clone.max_messages == 8
        assert clone.max_tokens == 1000
        assert clone.validation_retries == 2
        assert clone.on_tool_call is agent.on_tool_call
        assert clone.validator is agent.validator

    def test_the_clone_has_its_own_group_list(self):
        agent = Agent(api_key="k", tools=[search], enabled_groups=["web"])
        clone = agent._clone()

        clone.enable_group("extra")

        assert agent.enabled_groups == ["web"]

    def test_the_clone_shares_the_connection_pool(self):
        agent = Agent(api_key="k")
        assert agent._clone().client is agent.client

    def test_the_clone_starts_with_no_history(self):
        agent = Agent(api_key="k")
        agent.messages = [Message(role="user", content="old")]

        assert agent._clone().messages == []


class TestBatch:
    async def test_every_prompt_is_answered(self, make_agent):
        agent, fake = make_agent()
        for i in range(3):
            fake.queue(openai_response(f"answer {i}"))

        results = await agent.batch(["a", "b", "c"])

        assert sorted(results) == ["answer 0", "answer 1", "answer 2"]

    async def test_usage_counts_real_requests(self, make_agent):
        agent, fake = make_agent(tools=[add])
        # One prompt that needs a tool step, so two API calls for one prompt.
        fake.queue(openai_response(None, [openai_tool_call("add", {"a": 1, "b": 1})]))
        fake.queue(openai_response("two"))

        await agent.batch(["add one and one"])

        assert agent.usage.requests == 2

    async def test_concurrency_is_bounded(self, make_agent):
        agent, fake = make_agent()
        for _ in range(6):
            fake.queue(openai_response("ok"))

        in_flight = 0
        peak = 0
        original = agent.client.chat

        async def counting_chat(*args, **kwargs):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                await asyncio.sleep(0)
                return await original(*args, **kwargs)
            finally:
                in_flight -= 1

        agent.client.chat = counting_chat

        await agent.batch([f"q{i}" for i in range(6)], max_concurrency=2)

        assert peak <= 2

    async def test_batch_does_not_touch_the_parent_history(self, make_agent):
        agent, fake = make_agent()
        for _ in range(2):
            fake.queue(openai_response("ok"))

        await agent.batch(["a", "b"])

        assert agent.messages == []


class TestMaxSteps:
    async def test_exhausting_steps_raises(self, make_agent):
        agent, fake = make_agent(tools=[add], max_steps=3)
        for _ in range(3):
            fake.queue(
                openai_response(None, [openai_tool_call("add", {"a": 1, "b": 1})])
            )

        with pytest.raises(MaxStepsError) as excinfo:
            await agent.run("loop")

        assert excinfo.value.steps == 3
        assert "max_steps" in str(excinfo.value)

    async def test_the_partial_answer_is_available(self, make_agent):
        agent, fake = make_agent(tools=[add], max_steps=1)
        fake.queue(
            openai_response(
                "Working on it...", [openai_tool_call("add", {"a": 1, "b": 1})]
            )
        )

        with pytest.raises(MaxStepsError) as excinfo:
            await agent.run("loop")

        assert excinfo.value.partial == "Working on it..."


class TestSessionIds:
    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = JSONMemory(tmpdir)

            with pytest.raises(ValueError, match="Invalid session id"):
                memory.save("../../evil", [Message(role="user", content="x")])

    @pytest.mark.parametrize(
        "session_id", ["../evil", "a/b", "..", ".hidden", "", "with space", "a\\b"]
    )
    def test_unsafe_ids_are_rejected(self, session_id):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError):
                JSONMemory(tmpdir).load(session_id)

    @pytest.mark.parametrize("session_id", ["chat", "my-chat", "my_chat.v2", "a1"])
    def test_ordinary_ids_are_accepted(self, session_id):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = JSONMemory(tmpdir)
            memory.save(session_id, [Message(role="user", content="x")])

            assert memory.load(session_id)[0].content == "x"

    def test_an_agent_rejects_an_unsafe_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError):
                Agent(api_key="k", session="../escape", memory=JSONMemory(tmpdir))


class TestAtomicWrites:
    def test_a_failed_write_leaves_the_old_file_intact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = JSONMemory(tmpdir)
            memory.save("chat", [Message(role="user", content="good")])
            path = Path(tmpdir) / "chat.json"

            with pytest.raises(TypeError):
                write_json_atomically(path, [object()])

            assert memory.load("chat")[0].content == "good"
            assert [p.name for p in Path(tmpdir).iterdir()] == ["chat.json"]

    def test_no_temp_files_are_left_behind(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = JSONMemory(tmpdir)
            memory.save("chat", [Message(role="user", content="x")])

            assert [p.name for p in Path(tmpdir).iterdir()] == ["chat.json"]

    def test_corrupt_session_files_report_clearly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "chat.json").write_text("{not json")

            with pytest.raises(ValueError, match="not valid JSON"):
                JSONMemory(tmpdir).load("chat")

    def test_agent_save_is_atomic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = Agent(api_key="k")
            agent.messages = [Message(role="user", content="hello")]
            path = Path(tmpdir) / "chat.json"

            agent.save(str(path))

            assert json.loads(path.read_text())[0]["content"] == "hello"
            assert [p.name for p in Path(tmpdir).iterdir()] == ["chat.json"]
