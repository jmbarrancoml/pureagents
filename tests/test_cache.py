"""Tests for the response cache."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pure_agents import clear_cache, set_cache_size, tool
from pure_agents.agent import _response_cache
from tests.conftest import openai_response, openai_tool_call


@pytest.fixture(autouse=True)
def clean_cache():
    clear_cache()
    set_cache_size(128)
    yield
    clear_cache()
    set_cache_size(128)


@tool
def search(query: str) -> str:
    """Search the web."""
    return f"results for {query}"


class TestCacheKey:
    async def test_repeated_prompt_hits_the_cache(self, make_agent):
        agent, fake = make_agent(cache=True)
        fake.queue(openai_response("cached answer"))

        first = await agent.run("hello")
        agent.clear()
        second = await agent.run("hello")

        assert first == second == "cached answer"
        assert fake.call_count == 1

    async def test_different_system_prompts_do_not_share_an_entry(self, make_agent):
        pirate, pirate_api = make_agent(cache=True, system_prompt="You are a pirate.")
        lawyer, lawyer_api = make_agent(cache=True, system_prompt="You are a lawyer.")
        pirate_api.queue(openai_response("Arr."))
        lawyer_api.queue(openai_response("Per my last email."))

        assert await pirate.run("hello") == "Arr."
        assert await lawyer.run("hello") == "Per my last email."

    async def test_different_providers_do_not_share_an_entry(self, make_agent):
        openai_agent, openai_api = make_agent(cache=True, system_prompt="x")
        mistral_agent, mistral_api = make_agent(
            provider="mistral", cache=True, system_prompt="x"
        )
        openai_agent.model = mistral_agent.model = "same-model"
        openai_api.queue(openai_response("from openai"))
        mistral_api.queue(openai_response("from mistral"))

        assert await openai_agent.run("hello") == "from openai"
        assert await mistral_agent.run("hello") == "from mistral"

    async def test_different_tool_sets_do_not_share_an_entry(self, make_agent):
        plain, plain_api = make_agent(cache=True, system_prompt="x")
        armed, armed_api = make_agent(cache=True, system_prompt="x", tools=[search])
        plain_api.queue(openai_response("no tools"))
        armed_api.queue(openai_response("with tools"))

        assert await plain.run("hello") == "no tools"
        assert await armed.run("hello") == "with tools"


class TestCacheBehaviour:
    async def test_hit_keeps_the_conversation_consistent(self, make_agent):
        agent, fake = make_agent(cache=True)
        fake.queue(openai_response("cached answer"))
        fake.queue(openai_response("follow up"))

        await agent.run("hello")
        agent.messages = agent.messages[:1]  # keep only the system message
        await agent.run("hello")  # served from cache
        await agent.run("and now?")

        roles = [m["role"] for m in fake.last_request()["messages"]]
        assert roles == ["system", "user", "assistant", "user"]

    async def test_hit_returns_the_requested_dataclass(self, make_agent):
        @dataclass
        class Analysis:
            sentiment: str

        agent, fake = make_agent(cache=True)
        fake.queue(openai_response('{"sentiment": "positive"}'))

        first = await agent.run("review this", output=Analysis)
        agent.clear()
        second = await agent.run("review this", output=Analysis)

        assert first == second == Analysis(sentiment="positive")
        assert isinstance(second, Analysis)
        assert fake.call_count == 1

    async def test_failed_validation_is_not_cached(self, make_agent):
        agent, fake = make_agent(cache=True, validator=lambda text: text == "good")
        fake.queue(openai_response("bad"))
        fake.queue(openai_response("bad again"))

        await agent.run("hello")
        agent.clear()
        await agent.run("hello")

        assert fake.call_count == 2

    async def test_cache_is_bounded(self, make_agent):
        set_cache_size(2)
        agent, fake = make_agent(cache=True)
        for i in range(4):
            fake.queue(openai_response(f"answer {i}"))
            agent.clear()
            await agent.run(f"prompt {i}")

        assert len(_response_cache) == 2

    async def test_tool_runs_are_not_replayed_from_a_stale_key(self, make_agent):
        agent, fake = make_agent(cache=True, tools=[search])
        fake.queue(openai_response(None, [openai_tool_call("search", {"query": "x"})]))
        fake.queue(openai_response("done"))

        assert await agent.run("look it up") == "done"
        assert fake.call_count == 2
