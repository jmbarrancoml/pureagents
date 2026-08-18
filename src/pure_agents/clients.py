"""LLM provider clients."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from typing import Any

import httpx

from pure_agents.message import Message
from pure_agents.tool import Tool

DIALECTS = ("openai", "anthropic")

# Sent when a provider declares no environment variable, which is the shape of
# a local server. Ollama, LM Studio and vLLM accept any bearer token.
PLACEHOLDER_KEY = "not-needed"


@dataclass(frozen=True)
class Provider:
    """Where to send requests, and in which wire format."""

    base_url: str
    env_var: str | None = None
    default_model: str | None = None
    dialect: str = "openai"
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def needs_key(self) -> bool:
        return self.env_var is not None

    def with_headers(self, extra: dict[str, str] | None) -> Provider:
        if not extra:
            return self
        return replace(self, headers={**self.headers, **extra})


# Built-in providers. Their default models are the cheapest current-generation
# tier each one offers that still drives a tool loop reliably; override with
# Agent(model=...). Third-party endpoints are deliberately absent: their model
# names move faster than this file can track, so register them yourself.
PROVIDERS: dict[str, Provider] = {
    "mistral": Provider(
        base_url="https://api.mistral.ai/v1",
        env_var="MISTRAL_API_KEY",
        default_model="mistral-large-latest",
    ),
    "openai": Provider(
        base_url="https://api.openai.com/v1",
        env_var="OPENAI_API_KEY",
        default_model="gpt-5.6-luna",
    ),
    "anthropic": Provider(
        base_url="https://api.anthropic.com/v1",
        env_var="ANTHROPIC_API_KEY",
        default_model="claude-sonnet-5",
        dialect="anthropic",
    ),
}


def register_provider(
    name: str,
    *,
    base_url: str,
    env_var: str | None = None,
    default_model: str | None = None,
    dialect: str = "openai",
    headers: dict[str, str] | None = None,
    overwrite: bool = False,
) -> Provider:
    """Teach Agent about another endpoint, then use it by name.

        register_provider("ollama", base_url="http://localhost:11434/v1")
        agent = Agent(provider="ollama", model="llama3.3")

    Leave env_var unset for a local server that does not check the key. Set
    default_model only if you are willing to keep it current; otherwise callers
    pass model= and get a clear error when they forget.
    """
    if not name or not name.strip():
        raise ValueError("Provider name cannot be empty.")
    if name in PROVIDERS and not overwrite:
        raise ValueError(
            f"Provider {name!r} is already registered. "
            "Pass overwrite=True to replace it."
        )
    if not base_url.startswith(("http://", "https://")):
        raise ValueError(
            f"base_url must start with http:// or https://, got {base_url!r}"
        )
    if dialect not in DIALECTS:
        raise ValueError(f"Unknown dialect {dialect!r}. Use one of {list(DIALECTS)}.")

    provider = Provider(
        base_url=base_url.rstrip("/"),
        env_var=env_var,
        default_model=default_model,
        dialect=dialect,
        headers=dict(headers or {}),
    )
    PROVIDERS[name] = provider
    return provider


BUILTIN_PROVIDERS = frozenset(PROVIDERS)


def unregister_provider(name: str) -> None:
    """Remove a registered provider. Built-ins cannot be removed."""
    if name in BUILTIN_PROVIDERS:
        raise ValueError(f"Cannot unregister the built-in provider {name!r}.")
    PROVIDERS.pop(name, None)


def _with_images(
    messages: list[Message],
    images: list[tuple[str, str]] | None,
) -> list[dict[str, Any]]:
    """Serialise messages, attaching images to the last user turn."""
    serialised = [m.to_dict() for m in messages]
    if not images:
        return serialised

    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role != "user":
            continue
        content: list[dict[str, Any]] = [
            {"type": "text", "text": messages[index].content}
        ]
        for data, media_type in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{data}"},
                }
            )
        serialised[index]["content"] = content
        break

    return serialised


@dataclass
class LLMClient:
    """Simple LLM API client. Works with OpenAI-compatible APIs."""

    api_key: str
    base_url: str = "https://api.mistral.ai/v1"
    timeout: float = 60.0
    # Optional here: OpenAI-compatible APIs pick their own ceiling.
    max_tokens: int | None = None
    last_input_tokens: int = 0
    last_output_tokens: int = 0
    # Extra headers the endpoint needs, e.g. an OpenRouter attribution header.
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Injectable for tests; None uses httpx's real network transport.
    transport: httpx.AsyncBaseTransport | None = None

    # Reused across requests so connections are pooled and TLS is negotiated
    # once, instead of a fresh handshake per model call.
    _client: httpx.AsyncClient | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout, transport=self.transport
            )
        return self._client

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

    async def aclose(self) -> None:
        """Close the pooled connections. Safe to call more than once."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def chat(
        self,
        model: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        images: list[tuple[str, str]] | None = None,
    ) -> Message:
        # Reset first: a response without a usage block used to leave the
        # previous call's numbers in place, which Agent then counted twice.
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        client = self._http()
        payload: dict[str, Any] = {
            "model": model,
            "messages": _with_images(messages, images),
        }

        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        if tools:
            payload["tools"] = [t.to_dict() for t in tools]
            payload["tool_choice"] = tool_choice or "auto"

        response = await client.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        if "usage" in data:
            self.last_input_tokens = data["usage"].get("prompt_tokens", 0)
            self.last_output_tokens = data["usage"].get("completion_tokens", 0)

        choice = data["choices"][0]["message"]
        return Message(
            role="assistant",
            content=choice.get("content") or "",
            tool_calls=choice.get("tool_calls", []),
        )

    async def chat_stream(
        self,
        model: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        images: list[tuple[str, str]] | None = None,
    ) -> AsyncIterator[tuple[str, Message | None]]:
        # Reset first: a response without a usage block used to leave the
        # previous call's numbers in place, which Agent then counted twice.
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        client = self._http()
        payload: dict[str, Any] = {
            "model": model,
            "messages": _with_images(messages, images),
            "stream": True,
            # Without this a streamed call reports no tokens at all.
            "stream_options": {"include_usage": True},
        }

        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        if tools:
            payload["tools"] = [t.to_dict() for t in tools]
            payload["tool_choice"] = tool_choice or "auto"

        async with client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
        ) as response:
            response.raise_for_status()

            content = ""
            tool_calls: list[dict[str, Any]] = []

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                data = json.loads(data_str)

                if data.get("usage"):
                    self.last_input_tokens = data["usage"].get("prompt_tokens", 0)
                    self.last_output_tokens = data["usage"].get("completion_tokens", 0)

                # The usage-only chunk carries an empty choices list.
                if not data.get("choices"):
                    continue

                delta = data["choices"][0].get("delta", {})

                if delta.get("content"):
                    content += delta["content"]
                    yield delta["content"], None

                if delta.get("tool_calls"):
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        while len(tool_calls) <= idx:
                            tool_calls.append(
                                {
                                    "id": "",
                                    "function": {"name": "", "arguments": ""},
                                }
                            )
                        if tc.get("id"):
                            tool_calls[idx]["id"] = tc["id"]
                        fn = tc.get("function", {})
                        if fn.get("name"):
                            tool_calls[idx]["function"]["name"] = fn["name"]
                        if fn.get("arguments"):
                            tool_calls[idx]["function"]["arguments"] += fn["arguments"]

            yield (
                "",
                Message(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls if tool_calls else [],
                ),
            )


class TruncatedResponseError(RuntimeError):
    """The provider stopped mid-response, leaving unusable output."""


def _is_tool_result_message(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "user"
        and isinstance(content, list)
        and bool(content)
        and content[0].get("type") == "tool_result"
    )


DEFAULT_MAX_TOKENS = 4096


@dataclass
class AnthropicClient:
    """Anthropic API client."""

    api_key: str
    base_url: str = "https://api.anthropic.com/v1"
    timeout: float = 60.0
    max_tokens: int = DEFAULT_MAX_TOKENS
    last_input_tokens: int = 0
    last_output_tokens: int = 0
    # Extra headers the endpoint needs, e.g. an OpenRouter attribution header.
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Injectable for tests; None uses httpx's real network transport.
    transport: httpx.AsyncBaseTransport | None = None

    def _convert_messages(
        self,
        messages: list[Message],
        images: list[tuple[str, str]] | None = None,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        system_prompt = None
        converted = []

        for i, msg in enumerate(messages):
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                # Add images to last user message
                if images and i == len(messages) - 1:
                    content: list[dict[str, Any]] = []
                    for img_data, media_type in images:
                        content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": img_data,
                                },
                            }
                        )
                    content.append({"type": "text", "text": msg.content})
                    converted.append({"role": "user", "content": content})
                else:
                    converted.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                if msg.tool_calls:
                    content = []
                    if msg.content:
                        content.append({"type": "text", "text": msg.content})
                    for tc in msg.tool_calls:
                        content.append(
                            {
                                "type": "tool_use",
                                "id": tc["id"],
                                "name": tc["function"]["name"],
                                "input": json.loads(tc["function"]["arguments"]),
                            }
                        )
                    converted.append({"role": "assistant", "content": content})
                else:
                    converted.append({"role": "assistant", "content": msg.content})
            elif msg.role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content,
                }
                # Anthropic requires alternating roles, so every result from one
                # assistant turn goes into a single user message. Emitting one
                # message per result made parallel tool calls a 400.
                if converted and _is_tool_result_message(converted[-1]):
                    converted[-1]["content"].append(block)
                else:
                    converted.append({"role": "user", "content": [block]})

        return system_prompt, converted

    def _convert_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.schema(),
            }
            for t in tools
        ]

    # Reused across requests so connections are pooled and TLS is negotiated
    # once, instead of a fresh handshake per model call.
    _client: httpx.AsyncClient | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout, transport=self.transport
            )
        return self._client

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

    async def aclose(self) -> None:
        """Close the pooled connections. Safe to call more than once."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def chat(
        self,
        model: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        images: list[tuple[str, str]] | None = None,
    ) -> Message:
        system_prompt, converted_msgs = self._convert_messages(messages, images)

        # Reset first: a response without a usage block used to leave the
        # previous call's numbers in place, which Agent then counted twice.
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        client = self._http()
        payload: dict[str, Any] = {
            "model": model,
            "messages": converted_msgs,
            "max_tokens": self.max_tokens,
        }

        if system_prompt:
            payload["system"] = system_prompt

        if tools:
            payload["tools"] = self._convert_tools(tools)
            if tool_choice == "required":
                payload["tool_choice"] = {"type": "any"}
            elif tool_choice == "none":
                payload["tool_choice"] = {"type": "none"}

        response = await client.post(
            f"{self.base_url}/messages",
            headers=self._headers(),
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        if "usage" in data:
            self.last_input_tokens = data["usage"].get("input_tokens", 0)
            self.last_output_tokens = data["usage"].get("output_tokens", 0)

        content = ""
        tool_calls = []

        for block in data.get("content", []):
            if block["type"] == "text":
                content += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append(
                    {
                        "id": block["id"],
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block["input"]),
                        },
                    }
                )

        # A response cut off at max_tokens leaves tool_use input truncated, so
        # say so instead of handing the agent a half-formed call.
        if data.get("stop_reason") == "max_tokens":
            if tool_calls:
                raise TruncatedResponseError(
                    f"Anthropic hit max_tokens ({self.max_tokens}) mid tool call. "
                    "Raise Agent(max_tokens=...)."
                )
            content += f"\n\n[Response truncated at max_tokens={self.max_tokens}]"

        return Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        )

    async def chat_stream(
        self,
        model: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        images: list[tuple[str, str]] | None = None,
    ) -> AsyncIterator[tuple[str, Message | None]]:
        system_prompt, converted_msgs = self._convert_messages(messages, images)

        # Reset first: a response without a usage block used to leave the
        # previous call's numbers in place, which Agent then counted twice.
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        client = self._http()
        payload: dict[str, Any] = {
            "model": model,
            "messages": converted_msgs,
            "max_tokens": self.max_tokens,
            "stream": True,
        }

        if system_prompt:
            payload["system"] = system_prompt

        if tools:
            payload["tools"] = self._convert_tools(tools)
            if tool_choice == "required":
                payload["tool_choice"] = {"type": "any"}

        async with client.stream(
            "POST",
            f"{self.base_url}/messages",
            headers=self._headers(),
            json=payload,
        ) as response:
            response.raise_for_status()

            content = ""
            tool_calls: list[dict[str, Any]] = []
            current_tool: dict[str, Any] | None = None

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if not data_str or data_str == "[DONE]":
                    continue

                data = json.loads(data_str)
                event_type = data.get("type", "")

                if event_type == "message_start":
                    usage = data.get("message", {}).get("usage", {})
                    self.last_input_tokens = usage.get("input_tokens", 0)
                    self.last_output_tokens = usage.get("output_tokens", 0)
                elif event_type == "message_delta":
                    usage = data.get("usage", {})
                    if "output_tokens" in usage:
                        self.last_output_tokens = usage["output_tokens"]
                elif event_type == "content_block_start":
                    block = data.get("content_block", {})
                    if block.get("type") == "tool_use":
                        current_tool = {
                            "id": block["id"],
                            "function": {
                                "name": block["name"],
                                "arguments": "",
                            },
                        }
                elif event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        content += text
                        yield text, None
                    elif delta.get("type") == "input_json_delta":
                        if current_tool:
                            current_tool["function"]["arguments"] += delta.get(
                                "partial_json", ""
                            )
                elif event_type == "content_block_stop":
                    if current_tool:
                        if not current_tool["function"]["arguments"]:
                            current_tool["function"]["arguments"] = "{}"
                        tool_calls.append(current_tool)
                        current_tool = None

            yield (
                "",
                Message(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                ),
            )
