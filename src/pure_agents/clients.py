"""LLM provider clients."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from pure_agents.message import Message
from pure_agents.tool import Tool

PROVIDERS = {
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "env_var": "MISTRAL_API_KEY",
        "default_model": "mistral-large-latest",
        "client": "openai",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_var": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
        "client": "openai",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "env_var": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-4-20250514",
        "client": "anthropic",
    },
}


@dataclass
class LLMClient:
    """Simple LLM API client. Works with OpenAI-compatible APIs."""

    api_key: str
    base_url: str = "https://api.mistral.ai/v1"
    timeout: float = 60.0
    last_input_tokens: int = 0
    last_output_tokens: int = 0

    async def chat(
        self,
        model: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        images: list[tuple[str, str]] | None = None,
    ) -> Message:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Convert messages, adding images to last user message
            msg_list = []
            for i, m in enumerate(messages):
                msg_dict = m.to_dict()
                # Add images to the last user message
                if images and m.role == "user" and i == len(messages) - 1:
                    content = [{"type": "text", "text": m.content}]
                    for img_data, media_type in images:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{img_data}"
                            },
                        })
                    msg_dict["content"] = content
                msg_list.append(msg_dict)

            payload: dict[str, Any] = {
                "model": model,
                "messages": msg_list,
            }

            if tools:
                payload["tools"] = [t.to_dict() for t in tools]
                payload["tool_choice"] = tool_choice or "auto"

            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
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
    ) -> AsyncIterator[tuple[str, Message | None]]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            payload: dict[str, Any] = {
                "model": model,
                "messages": [m.to_dict() for m in messages],
                "stream": True,
            }

            if tools:
                payload["tools"] = [t.to_dict() for t in tools]
                payload["tool_choice"] = tool_choice or "auto"

            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
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
                    delta = data["choices"][0].get("delta", {})

                    if delta.get("content"):
                        content += delta["content"]
                        yield delta["content"], None

                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            while len(tool_calls) <= idx:
                                tool_calls.append({
                                    "id": "",
                                    "function": {"name": "", "arguments": ""},
                                })
                            if tc.get("id"):
                                tool_calls[idx]["id"] = tc["id"]
                            fn = tc.get("function", {})
                            if fn.get("name"):
                                tool_calls[idx]["function"]["name"] = fn["name"]
                            if fn.get("arguments"):
                                tool_calls[idx]["function"]["arguments"] += fn[
                                    "arguments"
                                ]

                yield "", Message(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls if tool_calls else [],
                )


@dataclass
class AnthropicClient:
    """Anthropic API client."""

    api_key: str
    base_url: str = "https://api.anthropic.com/v1"
    timeout: float = 60.0
    last_input_tokens: int = 0
    last_output_tokens: int = 0

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
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": img_data,
                            },
                        })
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
                        content.append({
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": json.loads(tc["function"]["arguments"]),
                        })
                    converted.append({"role": "assistant", "content": content})
                else:
                    converted.append({"role": "assistant", "content": msg.content})
            elif msg.role == "tool":
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                    }],
                })

        return system_prompt, converted

    def _convert_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": {
                    "type": "object",
                    "properties": t.parameters,
                    "required": list(t.parameters.keys()),
                },
            }
            for t in tools
        ]

    async def chat(
        self,
        model: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        images: list[tuple[str, str]] | None = None,
    ) -> Message:
        system_prompt, converted_msgs = self._convert_messages(messages, images)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            payload: dict[str, Any] = {
                "model": model,
                "messages": converted_msgs,
                "max_tokens": 4096,
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
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
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
                tool_calls.append({
                    "id": block["id"],
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block["input"]),
                    },
                })

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
    ) -> AsyncIterator[tuple[str, Message | None]]:
        system_prompt, converted_msgs = self._convert_messages(messages)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            payload: dict[str, Any] = {
                "model": model,
                "messages": converted_msgs,
                "max_tokens": 4096,
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
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
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

                    if event_type == "content_block_start":
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
                            tool_calls.append(current_tool)
                            current_tool = None

                yield "", Message(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                )
