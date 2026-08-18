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

PROVIDER_ENV_VARS = [c.env_var for c in PROVIDERS.values() if c.env_var]


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """Keep a developer's real API keys out of the test run."""
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def clean_provider_registry():
    """Undo any register_provider() a test performs."""
    snapshot = dict(PROVIDERS)
    yield
    PROVIDERS.clear()
    PROVIDERS.update(snapshot)


class FakeLLM:
    """A scripted HTTP endpoint standing in for a provider API."""

    def __init__(self) -> None:
        self._queue: deque = deque()
        self.requests: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []
        self.urls: list[str] = []

    def queue(
        self,
        payload: dict[str, Any],
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> FakeLLM:
        def build(
            status_code=status_code, payload=payload, headers=headers or {}
        ) -> httpx.Response:
            return httpx.Response(status_code, json=payload, headers=headers)

        self._queue.append(build)
        return self

    def queue_sse(self, events: list[dict[str, Any] | str]) -> FakeLLM:
        """Queue a server-sent-event stream body."""
        body = "".join(
            f"data: {event if isinstance(event, str) else json.dumps(event)}\n\n"
            for event in events
        )

        def build(body=body) -> httpx.Response:
            return httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )

        self._queue.append(build)
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
            self.urls.append(str(request.url))
            if not self._queue:
                raise AssertionError(
                    f"FakeLLM received an unscripted request: {request.url}"
                )
            item = self._queue.popleft()
            if isinstance(item, Exception):
                raise item
            return item()

        return httpx.MockTransport(handler)


def openai_response(
    content: str | None = "",
    tool_calls: list[dict[str, Any]] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """Build an OpenAI-compatible chat completion payload."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message, "finish_reason": finish_reason}],
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


def openai_sse(
    text_chunks: list[str] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> list[dict[str, Any] | str]:
    """Build an OpenAI-compatible SSE stream."""
    events: list[dict[str, Any] | str] = []

    for chunk in text_chunks or []:
        events.append({"choices": [{"delta": {"content": chunk}}]})

    for index, call in enumerate(tool_calls or []):
        function = call["function"]
        events.append(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": call["id"],
                                    "function": {"name": function["name"]},
                                }
                            ]
                        }
                    }
                ]
            }
        )
        # Arguments arrive split across deltas, as real providers send them.
        raw = function["arguments"]
        midpoint = len(raw) // 2
        for piece in (raw[:midpoint], raw[midpoint:]):
            events.append(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": index, "function": {"arguments": piece}}
                                ]
                            }
                        }
                    ]
                }
            )

    events.append(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }
    )
    events.append("[DONE]")
    return events


def anthropic_sse(
    text_chunks: list[str] | None = None,
    tool_uses: list[dict[str, Any]] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> list[dict[str, Any] | str]:
    """Build an Anthropic SSE stream."""
    events: list[dict[str, Any] | str] = [
        {
            "type": "message_start",
            "message": {"usage": {"input_tokens": input_tokens, "output_tokens": 0}},
        }
    ]

    for chunk in text_chunks or []:
        events.append(
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": chunk},
            }
        )

    for use in tool_uses or []:
        events.append(
            {
                "type": "content_block_start",
                "content_block": {
                    "type": "tool_use",
                    "id": use["id"],
                    "name": use["name"],
                },
            }
        )
        events.append(
            {
                "type": "content_block_delta",
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(use.get("input", {})),
                },
            }
        )
        events.append({"type": "content_block_stop"})

    events.append({"type": "message_delta", "usage": {"output_tokens": output_tokens}})
    events.append({"type": "message_stop"})
    return events


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
