"""Test harness: scriptable fake LLM endpoints backed by httpx.MockTransport.

Nothing here touches the network. `FakeLLM` queues the responses an endpoint
will return, records every request body it receives, and hands out a transport
you can attach to an Agent's client.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import httpx
import pytest

from pure_agents import Agent
from pure_agents.clients import PROVIDERS

PROVIDER_ENV_VARS = [config["env_var"] for config in PROVIDERS.values()]


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """Keep a developer's real API keys out of the test run."""
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class FakeLLM:
    """A scripted HTTP endpoint standing in for a provider API."""

    def __init__(self) -> None:
        self._queue: deque = deque()
        self.requests: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []

    def queue(
        self,
        payload: dict[str, Any],
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> FakeLLM:
        self._queue.append((status_code, payload, headers or {}))
        return self

    def queue_error(
        self,
        status_code: int,
        headers: dict[str, str] | None = None,
    ) -> FakeLLM:
        return self.queue({"error": "boom"}, status_code, headers)

    def queue_exception(self, exc: Exception) -> FakeLLM:
        self._queue.append(exc)
        return self

    @property
    def call_count(self) -> int:
        return len(self.requests)

    @property
    def pending(self) -> int:
        return len(self._queue)

    def last_request(self) -> dict[str, Any]:
        return self.requests[-1]

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(json.loads(request.content))
            self.headers.append(request.headers)
            if not self._queue:
                raise AssertionError(
                    f"FakeLLM received an unscripted request: {request.url}"
                )
            item = self._queue.popleft()
            if isinstance(item, Exception):
                raise item
            status_code, payload, headers = item
            return httpx.Response(status_code, json=payload, headers=headers)

        return httpx.MockTransport(handler)


def openai_response(
    content: str | None = "",
    tool_calls: list[dict[str, Any]] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> dict[str, Any]:
    """Build an OpenAI-compatible chat completion payload."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


def openai_tool_call(
    name: str,
    arguments: Any = None,
    call_id: str = "call_1",
    raw_arguments: str | None = None,
) -> dict[str, Any]:
    """Build one tool_call entry. `raw_arguments` bypasses JSON encoding."""
    if raw_arguments is None:
        raw_arguments = json.dumps(arguments if arguments is not None else {})
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": raw_arguments},
    }


def anthropic_response(
    text: str = "",
    tool_uses: list[dict[str, Any]] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
    stop_reason: str = "end_turn",
) -> dict[str, Any]:
    """Build an Anthropic messages payload."""
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    for use in tool_uses or []:
        content.append({"type": "tool_use", **use})
    return {
        "content": content,
        "stop_reason": stop_reason,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def anthropic_tool_use(
    name: str,
    tool_input: dict[str, Any] | None = None,
    use_id: str = "toolu_1",
) -> dict[str, Any]:
    return {"id": use_id, "name": name, "input": tool_input or {}}


@pytest.fixture
def make_agent():
    """Build an Agent wired to a FakeLLM. Returns (agent, fake)."""

    def _make(provider: str = "openai", **kwargs) -> tuple[Agent, FakeLLM]:
        kwargs.setdefault("api_key", "test-key")
        agent = Agent(provider=provider, **kwargs)
        fake = FakeLLM()
        agent.client.transport = fake.transport()
        if agent.fallback_client is not None:
            agent.fallback_client.transport = fake.transport()
        return agent, fake

    return _make
