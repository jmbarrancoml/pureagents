"""Tests for tool timeouts and genuine parallel execution."""

from __future__ import annotations

import asyncio
import time

import pytest

from pure_agents import tool
from tests.conftest import openai_response, openai_tool_call


class TestTimeout:
    async def test_sync_tool_respects_its_timeout(self):
        @tool(timeout=0.2)
        def slow(x: str) -> str:
            """Blocks for a while."""
            time.sleep(1.5)
            return "finished"

        started = time.perf_counter()
        with pytest.raises(asyncio.TimeoutError):
            await slow.call(x="a")

        assert time.perf_counter() - started < 1.0

    async def test_async_tool_respects_its_timeout(self):
        @tool(timeout=0.2)
        async def slow(x: str) -> str:
            """Sleeps for a while."""
            await asyncio.sleep(1.5)
            return "finished"

        with pytest.raises(asyncio.TimeoutError):
            await slow.call(x="a")

    async def test_a_fast_tool_still_returns(self):
        @tool(timeout=2.0)
        def quick(x: str) -> str:
            """Returns immediately."""
            return f"got {x}"

        assert await quick.call(x="a") == "got a"

    async def test_timeout_reaches_the_model_as_a_tool_result(self, make_agent):
        @tool(timeout=0.2)
        def slow(x: str) -> str:
            """Blocks for a while."""
            time.sleep(1.5)
            return "finished"

        agent, fake = make_agent(tools=[slow])
        fake.queue(openai_response(None, [openai_tool_call("slow", {"x": "a"})]))
        fake.queue(openai_response("noted"))

        assert await agent.run("go") == "noted"
        assert "timed out" in fake.last_request()["messages"][-1]["content"]


class TestParallelism:
    async def test_sync_tools_do_not_block_the_event_loop(self):
        @tool
        def sleeper(label: str) -> str:
            """Blocks briefly."""
            time.sleep(0.3)
            return label

        started = time.perf_counter()
        results = await asyncio.gather(
            sleeper.call(label="a"),
            sleeper.call(label="b"),
            sleeper.call(label="c"),
        )
        elapsed = time.perf_counter() - started

        assert results == ["a", "b", "c"]
        # Serialised on the event loop this took ~0.9s.
        assert elapsed < 0.7

    async def test_agent_runs_parallel_sync_tools_concurrently(self, make_agent):
        @tool
        def sleeper(label: str) -> str:
            """Blocks briefly."""
            time.sleep(0.3)
            return label

        agent, fake = make_agent(tools=[sleeper])
        fake.queue(
            openai_response(
                None,
                [
                    openai_tool_call("sleeper", {"label": "a"}, call_id="c1"),
                    openai_tool_call("sleeper", {"label": "b"}, call_id="c2"),
                    openai_tool_call("sleeper", {"label": "c"}, call_id="c3"),
                ],
            )
        )
        fake.queue(openai_response("done"))

        started = time.perf_counter()
        await agent.run("run them all")
        elapsed = time.perf_counter() - started

        assert elapsed < 0.7

    async def test_async_tools_still_work(self):
        @tool
        async def fetch(url: str) -> str:
            """Pretends to fetch something."""
            await asyncio.sleep(0)
            return f"body of {url}"

        assert await fetch.call(url="x") == "body of x"
