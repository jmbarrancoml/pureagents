"""Tests for the retry policy and usage accounting."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from pure_agents import Agent, Usage
from pure_agents.agent import _is_retryable, _retry_delay
from tests.conftest import openai_response


def http_error(status_code: int, headers: dict[str, str] | None = None):
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(status_code, headers=headers or {}, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


class TestRetryPolicy:
    @pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 502, 503, 529])
    def test_transient_statuses_are_retryable(self, status):
        assert _is_retryable(http_error(status))

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_client_errors_are_not_retryable(self, status):
        assert not _is_retryable(http_error(status))

    def test_network_failures_are_retryable(self):
        assert _is_retryable(httpx.ConnectError("no route"))
        assert _is_retryable(asyncio.TimeoutError())

    def test_programming_errors_are_not_retryable(self):
        assert not _is_retryable(ValueError("bad code"))

    def test_retry_after_is_honoured(self):
        assert _retry_delay(http_error(429, {"retry-after": "7"}), 0) == 7.0

    def test_retry_after_is_capped(self):
        assert _retry_delay(http_error(429, {"retry-after": "9999"}), 0) == 60.0

    def test_unparseable_retry_after_falls_back_to_backoff(self):
        delay = _retry_delay(http_error(429, {"retry-after": "next tuesday"}), 2)
        assert 2.0 <= delay <= 4.0

    def test_backoff_grows_and_is_jittered(self):
        delays = {_retry_delay(http_error(500), 3) for _ in range(20)}
        assert len(delays) > 1
        assert all(4.0 <= d <= 8.0 for d in delays)


class TestRetryBehaviour:
    async def test_a_401_is_not_retried(self, make_agent):
        agent, fake = make_agent(retries=3)
        fake.queue_error(401)

        with pytest.raises(httpx.HTTPStatusError):
            await agent.run("hi")

        assert fake.call_count == 1

    async def test_a_500_is_retried(self, make_agent, monkeypatch):
        monkeypatch.setattr("pure_agents.agent._retry_delay", lambda exc, attempt: 0)
        agent, fake = make_agent(retries=2)
        fake.queue_error(500)
        fake.queue_error(500)
        fake.queue(openai_response("finally"))

        assert await agent.run("hi") == "finally"
        assert fake.call_count == 3

    async def test_retries_are_bounded(self, make_agent, monkeypatch):
        monkeypatch.setattr("pure_agents.agent._retry_delay", lambda exc, attempt: 0)
        agent, fake = make_agent(retries=2)
        for _ in range(3):
            fake.queue_error(503)

        with pytest.raises(httpx.HTTPStatusError):
            await agent.run("hi")

        assert fake.call_count == 3


class TestUsageAccounting:
    async def test_a_response_without_usage_is_not_double_counted(self, make_agent):
        agent, fake = make_agent()
        fake.queue(openai_response("one", prompt_tokens=10, completion_tokens=5))
        payload = openai_response("two")
        del payload["usage"]
        fake.queue(payload)

        await agent.run("a")
        await agent.run("b")

        assert agent.usage.input_tokens == 10
        assert agent.usage.output_tokens == 5


class TestCost:
    def test_known_model_is_priced(self):
        usage = Usage(model="claude-opus-5")
        usage.add(1_000_000, 1_000_000)

        assert usage.cost() == pytest.approx(30.00)

    def test_unknown_model_returns_none_rather_than_a_guess(self):
        usage = Usage(model="some-model-we-do-not-price")
        usage.add(1_000_000, 1_000_000)

        assert usage.cost() is None

    def test_explicit_rates_win(self):
        usage = Usage(model="some-model-we-do-not-price")
        usage.add(1_000_000, 0)
        usage.set_rates(2.0, 6.0)

        assert usage.cost() == pytest.approx(2.0)

    def test_a_model_can_be_named_explicitly(self):
        usage = Usage()
        usage.add(1_000_000, 0)

        assert usage.cost("claude-haiku-4-5") == pytest.approx(1.0)

    def test_the_agent_prices_its_own_model(self):
        agent = Agent(api_key="k", provider="anthropic")
        agent.usage.add(1_000_000, 0)

        assert agent.usage.model == "claude-opus-5"
        assert agent.usage.cost() == pytest.approx(5.0)
