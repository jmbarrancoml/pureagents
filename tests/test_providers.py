"""Tests for the provider registry and ad-hoc OpenAI-compatible endpoints."""

from __future__ import annotations

import pytest

from pure_agents import Agent, Provider, register_provider, unregister_provider
from pure_agents.clients import (
    PLACEHOLDER_KEY,
    PROVIDERS,
    AnthropicClient,
    LLMClient,
)
from tests.conftest import FakeLLM, anthropic_response, openai_response


def wire(agent: Agent) -> FakeLLM:
    fake = FakeLLM()
    agent.client.transport = fake.transport()
    return fake


class TestRegisterProvider:
    def test_registers_and_is_usable_by_name(self):
        register_provider("ollama", base_url="http://localhost:11434/v1")

        agent = Agent(provider="ollama", model="llama3.3")

        assert agent.provider == "ollama"
        assert agent.client.base_url == "http://localhost:11434/v1"
        assert agent.model == "llama3.3"

    async def test_requests_go_to_the_registered_url(self):
        register_provider("ollama", base_url="http://localhost:11434/v1")
        agent = Agent(provider="ollama", model="llama3.3")
        fake = wire(agent)
        fake.queue(openai_response("hi from llama"))

        assert await agent.run("hello") == "hi from llama"
        assert fake.urls[0] == "http://localhost:11434/v1/chat/completions"

    def test_a_default_model_is_optional_but_honoured(self):
        register_provider(
            "groq", base_url="https://api.groq.com/openai/v1", default_model="fast-1"
        )

        assert Agent(provider="groq", api_key="k").model == "fast-1"

    def test_no_default_model_means_model_is_required(self):
        register_provider("groq", base_url="https://api.groq.com/openai/v1")

        with pytest.raises(ValueError, match="has no default model"):
            Agent(provider="groq", api_key="k")

    def test_a_keyless_provider_needs_no_key(self):
        register_provider("ollama", base_url="http://localhost:11434/v1")

        agent = Agent(provider="ollama", model="llama3.3")

        assert agent.api_key == PLACEHOLDER_KEY

    def test_a_provider_with_an_env_var_reads_it(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "groq-key")
        register_provider(
            "groq", base_url="https://api.groq.com/openai/v1", env_var="GROQ_API_KEY"
        )

        assert Agent(provider="groq", model="m").client.api_key == "groq-key"

    def test_a_provider_with_an_env_var_still_requires_one(self):
        register_provider(
            "groq", base_url="https://api.groq.com/openai/v1", env_var="GROQ_API_KEY"
        )

        with pytest.raises(ValueError, match="API key required"):
            Agent(provider="groq", model="m")

    def test_the_anthropic_dialect_can_be_selected(self):
        register_provider(
            "bedrock-ish",
            base_url="https://example.test/v1",
            dialect="anthropic",
            default_model="claude-ish",
        )

        agent = Agent(provider="bedrock-ish", api_key="k")

        assert isinstance(agent.client, AnthropicClient)

    async def test_the_anthropic_dialect_speaks_that_wire_format(self):
        register_provider(
            "proxy", base_url="https://example.test/v1", dialect="anthropic"
        )
        agent = Agent(provider="proxy", model="claude-ish", api_key="k")
        fake = wire(agent)
        fake.queue(anthropic_response("routed"))

        assert await agent.run("hi") == "routed"
        assert fake.urls[0] == "https://example.test/v1/messages"

    def test_a_trailing_slash_is_trimmed(self):
        provider = register_provider("x", base_url="https://example.test/v1/")

        assert provider.base_url == "https://example.test/v1"


class TestRegisterProviderValidation:
    def test_a_duplicate_name_is_refused(self):
        register_provider("dup", base_url="https://example.test/v1")

        with pytest.raises(ValueError, match="already registered"):
            register_provider("dup", base_url="https://other.test/v1")

    def test_overwrite_replaces_it(self):
        register_provider("dup", base_url="https://example.test/v1")
        register_provider("dup", base_url="https://other.test/v1", overwrite=True)

        assert PROVIDERS["dup"].base_url == "https://other.test/v1"

    def test_a_built_in_cannot_be_shadowed_by_accident(self):
        with pytest.raises(ValueError, match="already registered"):
            register_provider("openai", base_url="https://evil.test/v1")

    @pytest.mark.parametrize("bad", ["example.test/v1", "ftp://example.test", ""])
    def test_the_base_url_must_be_http(self, bad):
        with pytest.raises(ValueError, match="must start with http"):
            register_provider("x", base_url=bad)

    def test_an_unknown_dialect_is_refused(self):
        with pytest.raises(ValueError, match="Unknown dialect"):
            register_provider("x", base_url="https://e.test/v1", dialect="gemini")

    def test_an_empty_name_is_refused(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            register_provider("  ", base_url="https://e.test/v1")


class TestUnregisterProvider:
    def test_removes_a_registered_provider(self):
        register_provider("temp", base_url="https://e.test/v1")
        unregister_provider("temp")

        assert "temp" not in PROVIDERS

    def test_built_ins_are_protected(self):
        with pytest.raises(ValueError, match="built-in"):
            unregister_provider("openai")

    def test_removing_an_unknown_name_is_a_no_op(self):
        unregister_provider("never-existed")


class TestAdHocBaseUrl:
    async def test_requests_go_to_the_given_url(self):
        agent = Agent(base_url="http://localhost:8000/v1", model="local-model")
        fake = wire(agent)
        fake.queue(openai_response("local answer"))

        assert await agent.run("hi") == "local answer"
        assert fake.urls[0] == "http://localhost:8000/v1/chat/completions"

    def test_it_needs_no_key(self):
        agent = Agent(base_url="http://localhost:8000/v1", model="m")

        assert agent.api_key == PLACEHOLDER_KEY
        assert isinstance(agent.client, LLMClient)

    def test_an_explicit_key_is_still_used(self):
        agent = Agent(base_url="http://localhost:8000/v1", model="m", api_key="secret")

        assert agent.client.api_key == "secret"

    def test_a_model_is_required(self):
        with pytest.raises(ValueError, match="has no default model"):
            Agent(base_url="http://localhost:8000/v1")

    def test_it_cannot_be_combined_with_provider(self):
        with pytest.raises(ValueError, match="not both"):
            Agent(provider="openai", base_url="http://localhost:8000/v1", model="m")

    @pytest.mark.parametrize("bad", ["localhost:8000", "ftp://x.test"])
    def test_the_url_must_be_http(self, bad):
        with pytest.raises(ValueError, match="must start with http"):
            Agent(base_url=bad, model="m")

    def test_a_trailing_slash_is_trimmed(self):
        agent = Agent(base_url="http://localhost:8000/v1/", model="m")

        assert agent.client.base_url == "http://localhost:8000/v1"


class TestExtraHeaders:
    async def test_agent_headers_reach_the_request(self):
        agent = Agent(
            base_url="https://openrouter.test/api/v1",
            model="m",
            api_key="k",
            headers={"HTTP-Referer": "https://myapp.test", "X-Title": "My App"},
        )
        fake = wire(agent)
        fake.queue(openai_response("ok"))

        await agent.run("hi")

        assert fake.headers[0]["x-title"] == "My App"
        assert fake.headers[0]["http-referer"] == "https://myapp.test"
        assert fake.headers[0]["authorization"] == "Bearer k"

    async def test_registered_headers_reach_the_request(self):
        register_provider(
            "openrouter",
            base_url="https://openrouter.test/api/v1",
            headers={"X-Title": "pureagents"},
        )
        agent = Agent(provider="openrouter", model="m")
        fake = wire(agent)
        fake.queue(openai_response("ok"))

        await agent.run("hi")

        assert fake.headers[0]["x-title"] == "pureagents"

    async def test_agent_headers_win_over_registered_ones(self):
        register_provider(
            "openrouter",
            base_url="https://openrouter.test/api/v1",
            headers={"X-Title": "default"},
        )
        agent = Agent(provider="openrouter", model="m", headers={"X-Title": "mine"})
        fake = wire(agent)
        fake.queue(openai_response("ok"))

        await agent.run("hi")

        assert fake.headers[0]["x-title"] == "mine"


class TestErrorMessages:
    def test_an_unknown_provider_lists_the_registered_ones(self):
        register_provider("ollama", base_url="http://localhost:11434/v1")

        with pytest.raises(ValueError, match="Unknown provider") as excinfo:
            Agent(provider="typo", api_key="k")

        message = str(excinfo.value)
        assert "ollama" in message
        assert "register_provider" in message


class TestCloning:
    def test_a_clone_keeps_an_ad_hoc_endpoint(self):
        agent = Agent(
            base_url="http://localhost:8000/v1",
            model="m",
            headers={"X-Title": "app"},
        )

        clone = agent._clone()

        assert clone.client.base_url == "http://localhost:8000/v1"
        assert clone.client.extra_headers == {"X-Title": "app"}

    async def test_batch_works_against_an_ad_hoc_endpoint(self):
        agent = Agent(base_url="http://localhost:8000/v1", model="m")
        fake = wire(agent)
        fake.queue(openai_response("a")).queue(openai_response("b"))

        assert sorted(await agent.batch(["one", "two"])) == ["a", "b"]

    def test_a_clone_keeps_a_registered_provider(self):
        register_provider("ollama", base_url="http://localhost:11434/v1")
        agent = Agent(provider="ollama", model="llama3.3")

        assert agent._clone().provider == "ollama"


class TestProviderRecord:
    def test_it_is_immutable(self):
        with pytest.raises(Exception):
            PROVIDERS["openai"].base_url = "https://evil.test"

    def test_needs_key_follows_env_var(self):
        assert Provider(base_url="https://e.test", env_var="X").needs_key
        assert not Provider(base_url="https://e.test").needs_key

    def test_built_in_defaults_are_unchanged(self):
        assert PROVIDERS["openai"].default_model == "gpt-5.6-luna"
        assert PROVIDERS["anthropic"].dialect == "anthropic"
        assert PROVIDERS["mistral"].env_var == "MISTRAL_API_KEY"
