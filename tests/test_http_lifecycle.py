"""Tests for HTTP client reuse and timeout propagation."""

from __future__ import annotations

import httpx

from pure_agents import Agent
from tests.conftest import FakeLLM, openai_response


class TestConnectionReuse:
    async def test_the_same_http_client_serves_every_request(self, make_agent):
        agent, fake = make_agent()
        for _ in range(3):
            fake.queue(openai_response("ok"))

        await agent.run("a")
        first = agent.client._http()
        await agent.run("b")
        await agent.run("c")

        assert agent.client._http() is first
        assert fake.call_count == 3

    async def test_aclose_releases_the_client(self, make_agent):
        agent, fake = make_agent()
        fake.queue(openai_response("ok"))
        await agent.run("a")

        opened = agent.client._http()
        await agent.aclose()

        assert opened.is_closed

    async def test_the_agent_works_again_after_being_closed(self, make_agent):
        agent, fake = make_agent()
        fake.queue(openai_response("first"))
        fake.queue(openai_response("second"))

        await agent.run("a")
        await agent.aclose()
        assert await agent.run("b") == "second"

    async def test_agent_is_an_async_context_manager(self):
        fake = FakeLLM()
        fake.queue(openai_response("ok"))

        async with Agent(api_key="k", provider="openai") as agent:
            agent.client.transport = fake.transport()
            assert await agent.run("hi") == "ok"
            opened = agent.client._http()

        assert opened.is_closed

    async def test_closing_also_closes_the_fallback(self):
        agent = Agent(api_key="k", fallback="anthropic", fallback_api_key="a")
        primary = agent.client._http()
        secondary = agent.fallback_client._http()

        await agent.aclose()

        assert primary.is_closed
        assert secondary.is_closed


class TestTimeoutPropagation:
    def test_agent_timeout_reaches_httpx(self):
        agent = Agent(api_key="k", timeout=300.0)

        assert agent.client.timeout == 300.0
        assert agent.client._http().timeout == httpx.Timeout(300.0)

    def test_default_timeout_when_none_is_set(self):
        agent = Agent(api_key="k")

        assert agent.client.timeout == 60.0

    def test_fallback_client_gets_the_same_timeout(self):
        agent = Agent(
            api_key="k",
            timeout=120.0,
            fallback="anthropic",
            fallback_api_key="a",
        )

        assert agent.fallback_client.timeout == 120.0
