"""Tests for API key resolution and provider fallback."""

from __future__ import annotations

import httpx
import pytest

from pure_agents import Agent
from tests.conftest import FakeLLM, anthropic_response, openai_response


class TestApiKeyPrecedence:
    def test_explicit_key_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "env-key")

        agent = Agent(api_key="explicit-key")

        assert agent.api_key == "explicit-key"
        assert agent.client.api_key == "explicit-key"

    def test_environment_is_used_when_no_key_is_passed(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "env-key")

        agent = Agent()

        assert agent.client.api_key == "env-key"

    def test_each_provider_reads_its_own_variable(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "mistral-key")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

        assert Agent(provider="openai").client.api_key == "openai-key"
        assert Agent(provider="mistral").client.api_key == "mistral-key"

    def test_missing_key_is_rejected(self):
        with pytest.raises(ValueError, match="API key required"):
            Agent()

    async def test_key_is_sent_as_a_bearer_token(self, make_agent):
        agent, fake = make_agent(api_key="secret")
        fake.queue(openai_response("ok"))

        await agent.run("hi")

        assert fake.headers[0]["authorization"] == "Bearer secret"

    async def test_anthropic_uses_the_x_api_key_header(self, make_agent):
        agent, fake = make_agent(provider="anthropic", api_key="secret")
        fake.queue(anthropic_response("ok"))

        await agent.run("hi")

        assert fake.headers[0]["x-api-key"] == "secret"
        assert fake.headers[0]["anthropic-version"] == "2023-06-01"


class TestFallback:
    def test_fallback_gets_its_own_key_and_model(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

        agent = Agent(api_key="mistral-key", provider="mistral", fallback="anthropic")

        assert agent.fallback_client.api_key == "anthropic-key"
        assert agent.fallback_model == "claude-opus-5"
        assert agent.model == "mistral-large-latest"

    def test_fallback_model_can_be_overridden(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

        agent = Agent(
            api_key="k",
            fallback="anthropic",
            fallback_model="claude-haiku-4-5",
        )

        assert agent.fallback_model == "claude-haiku-4-5"

    def test_missing_fallback_key_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="needs an API key"):
            Agent(api_key="k", fallback="anthropic")

    def test_unknown_fallback_provider_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown fallback provider"):
            Agent(api_key="k", fallback="nope")

    async def test_fallback_request_names_the_fallback_model(self):
        agent = Agent(
            api_key="mistral-key",
            provider="mistral",
            fallback="anthropic",
            fallback_api_key="anthropic-key",
        )

        primary, secondary = FakeLLM(), FakeLLM()
        agent.client.transport = primary.transport()
        agent.fallback_client.transport = secondary.transport()

        primary.queue_error(500)
        secondary.queue(anthropic_response("rescued"))

        assert await agent.run("hi") == "rescued"

        # The bug this guards: the primary's model used to be sent to the
        # fallback provider, which does not serve it.
        assert primary.last_request()["model"] == "mistral-large-latest"
        assert secondary.last_request()["model"] == "claude-opus-5"

    async def test_primary_is_used_when_it_succeeds(self):
        agent = Agent(
            api_key="k",
            fallback="anthropic",
            fallback_api_key="anthropic-key",
        )

        primary, secondary = FakeLLM(), FakeLLM()
        agent.client.transport = primary.transport()
        agent.fallback_client.transport = secondary.transport()

        primary.queue(openai_response("primary answered"))

        assert await agent.run("hi") == "primary answered"
        assert secondary.call_count == 0

    async def test_error_propagates_when_both_providers_fail(self):
        agent = Agent(
            api_key="k",
            fallback="anthropic",
            fallback_api_key="anthropic-key",
        )

        primary, secondary = FakeLLM(), FakeLLM()
        agent.client.transport = primary.transport()
        agent.fallback_client.transport = secondary.transport()

        primary.queue_error(500)
        secondary.queue_error(500)

        with pytest.raises(httpx.HTTPStatusError):
            await agent.run("hi")
