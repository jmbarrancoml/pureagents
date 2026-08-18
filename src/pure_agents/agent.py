"""Agent class and convenience functions."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import random
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import httpx

from pure_agents.clients import (
    PLACEHOLDER_KEY,
    PROVIDERS,
    AnthropicClient,
    LLMClient,
    Provider,
)
from pure_agents.memory import JSONMemory, Memory, write_json_atomically
from pure_agents.message import Message
from pure_agents.schema import dataclass_schema
from pure_agents.tool import Tool

T = TypeVar("T")

TEMPLATES: dict[str, str] = {
    "default": (
        "You are a helpful assistant. "
        "When you have the final answer, respond directly. Be concise."
    ),
    "coder": (
        "You are an expert programmer. Write clean, efficient code. "
        "Explain your reasoning briefly. Use best practices."
    ),
    "researcher": (
        "You are a research assistant. Provide accurate, well-sourced information. "
        "Be thorough but concise. Cite sources when possible."
    ),
    "creative": (
        "You are a creative writer. Be imaginative and engaging. "
        "Use vivid language and original ideas."
    ),
    "analyst": (
        "You are a data analyst. Provide clear, logical analysis. "
        "Use numbers and evidence. Be objective and precise."
    ),
    "tutor": (
        "You are a patient tutor. Explain concepts clearly and simply. "
        "Use examples. Adapt to the student's level."
    ),
}

DEFAULT_REQUEST_TIMEOUT = 60.0
MAX_RETRY_DELAY = 60.0

# Statuses worth trying again. A 400 or a 401 will fail identically every time,
# so retrying them only delays the error the caller needs to see.
RETRYABLE_STATUS = {408, 409, 425, 429}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status in RETRYABLE_STATUS or status >= 500
    return isinstance(exc, (httpx.TransportError, asyncio.TimeoutError))


def _retry_delay(exc: BaseException, attempt: int) -> float:
    """Back off, preferring the server's own Retry-After when it sends one."""
    if isinstance(exc, httpx.HTTPStatusError):
        retry_after = exc.response.headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after), MAX_RETRY_DELAY)
            except ValueError:
                pass
    backoff = min(2.0**attempt, MAX_RETRY_DELAY)
    # Equal jitter, so a fleet of agents does not retry in lockstep.
    return backoff / 2 + random.uniform(0, backoff / 2)


VALIDATION_RETRY_PROMPT = "Your response was invalid. Please try again."

DEFAULT_CACHE_SIZE = 128


class _ResponseCache:
    """Bounded LRU cache for whole-response reuse."""

    def __init__(self, maxsize: int = DEFAULT_CACHE_SIZE) -> None:
        self.maxsize = maxsize
        self._entries: OrderedDict[str, str] = OrderedDict()

    def get(self, key: str) -> str | None:
        if key not in self._entries:
            return None
        self._entries.move_to_end(key)
        return self._entries[key]

    def set(self, key: str, value: str) -> None:
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def resize(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("Cache size must be at least 1")
        self.maxsize = maxsize
        while len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)


_response_cache = _ResponseCache()


# List prices in USD per million tokens, checked against each provider's own
# pricing page on 2026-08-17. Rates move and models come and go, so a model
# that is not listed makes cost() return None rather than guess. Override any
# entry, or price an unlisted model, with Usage.set_rates().
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    # Introductory rate through 2026-08-31; the list price is 3.00 / 15.00.
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    # OpenAI
    "gpt-5.6-sol": (5.00, 30.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    # Mistral, under both the moving alias and the pinned id it resolves to
    "mistral-large-latest": (0.50, 1.50),
    "mistral-large-2512": (0.50, 1.50),
    "mistral-medium-latest": (1.50, 7.50),
    "mistral-medium-3505": (1.50, 7.50),
    "mistral-small-latest": (0.15, 0.60),
    "mistral-small-2603": (0.15, 0.60),
}


@dataclass
class Usage:
    """Token usage and cost tracking."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0
    model: str | None = None
    provider: str | None = None
    # Overrides the table above, as (input, output) USD per million tokens.
    rates: tuple[float, float] | None = None

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
        self.requests += 1

    def set_rates(self, input_per_million: float, output_per_million: float) -> None:
        """Price this usage with your own rates, in USD per million tokens."""
        self.rates = (input_per_million, output_per_million)

    def cost(self, model: str | None = None) -> float | None:
        """Estimated spend in USD, or None when the model's rate is unknown."""
        rates = self.rates or MODEL_PRICING.get(model or self.model or "")
        if rates is None:
            return None
        return (self.input_tokens / 1_000_000) * rates[0] + (
            self.output_tokens / 1_000_000
        ) * rates[1]

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.requests = 0


MAX_STEPS_MESSAGE = "Max steps reached without final answer."
DEFAULT_BATCH_CONCURRENCY = 5


class MaxStepsError(RuntimeError):
    """The agent used every step without producing a final answer."""

    def __init__(self, steps: int, partial: str = "") -> None:
        super().__init__(
            f"{MAX_STEPS_MESSAGE} Used all {steps} steps. Raise Agent(max_steps=...)."
        )
        self.steps = steps
        self.partial = partial


@dataclass
class StreamEvent:
    """One event from Agent.stream().

    type is "text" for a content chunk, "tool_call" when the model asks for a
    tool, "tool_result" once that tool has run, and "done" for the final answer.
    """

    type: str
    content: str = ""
    name: str | None = None
    id: str | None = None
    arguments: dict[str, Any] | None = None


def _safe_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


class StructuredOutputError(ValueError):
    """The model's reply could not be turned into the requested dataclass."""

    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


def _schema_from_dataclass(cls: type) -> dict[str, Any]:
    return dataclass_schema(cls)


def _extract_json(content: str) -> str:
    """Pull the JSON object out of a reply that may be fenced or chatty."""
    text = content.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines[1:]).strip()

    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]

    return text


def _parse_structured(content: str, output_cls: type[T]) -> T:
    text = _extract_json(content)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise StructuredOutputError(
            f"Model did not return valid JSON for {output_cls.__name__}: {e}",
            content,
        ) from e

    if not isinstance(data, dict):
        raise StructuredOutputError(
            f"Model returned {type(data).__name__}, not a JSON object, "
            f"for {output_cls.__name__}",
            content,
        )

    try:
        return output_cls(**data)
    except TypeError as e:
        raise StructuredOutputError(
            f"Model's JSON does not match {output_cls.__name__}: {e}",
            content,
        ) from e


def _cache_key(
    messages: list[Message],
    model: str,
    provider: str,
    tools: list[Tool],
    tool_choice: str | None,
) -> str:
    """Hash everything that can change the answer, not just the prompt."""
    payload = {
        "provider": provider,
        "model": model,
        "tool_choice": tool_choice,
        "tools": sorted(
            (t.to_dict() for t in tools), key=lambda d: d["function"]["name"]
        ),
        "messages": [m.to_dict() for m in messages],
    }
    serialised = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()


DEFAULT_PROVIDER = "mistral"
CUSTOM_PROVIDER = "custom"


def _resolve_provider(
    provider: str | None,
    base_url: str | None,
) -> tuple[str, Provider]:
    """Work out which endpoint to talk to, and in which wire format."""
    if base_url:
        if provider is not None:
            raise ValueError(
                "Pass either provider= or base_url=, not both. To reuse a custom "
                "endpoint by name, call register_provider() once instead."
            )
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(
                f"base_url must start with http:// or https://, got {base_url!r}"
            )
        # An ad-hoc endpoint always speaks the OpenAI dialect: that is what
        # "OpenAI-compatible" means. For anything else, register_provider().
        return CUSTOM_PROVIDER, Provider(base_url=base_url.rstrip("/"))

    name = provider or DEFAULT_PROVIDER
    if name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider: {name}. Registered: {sorted(PROVIDERS)}. "
            "Add your own with register_provider()."
        )
    return name, PROVIDERS[name]


def _key_from_env(config: Provider) -> str:
    """Read the provider's key, or a placeholder for a keyless local server."""
    if not config.needs_key:
        return PLACEHOLDER_KEY
    return os.environ.get(config.env_var or "", "")


def _load_image(path: str) -> tuple[str, str]:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_types.get(suffix, "image/png")
    data = base64.b64encode(path_obj.read_bytes()).decode()
    return data, media_type


class Agent:
    """
    A simple LLM agent that can use tools.

    Example:
        @tool
        def search(query: str) -> str:
            '''Search the web.'''
            return "Results..."

        agent = Agent(tools=[search])
        result = await agent.run("Find the weather in Madrid")
    """

    def __init__(
        self,
        model: str | None = None,
        tools: list[Tool] | None = None,
        api_key: str | None = None,
        max_steps: int = 10,
        max_tokens: int | None = None,
        system: str | None = None,
        template: str | None = None,
        debug: bool = False,
        provider: str | None = None,
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
        session: str | None = None,
        memory: Memory | None = None,
        # Hooks
        on_tool_call: Callable[[str, dict], None] | None = None,
        on_tool_result: Callable[[str, str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
        on_plan: Callable[[str], None] | None = None,
        # Reliability
        retries: int = 0,
        timeout: float | None = None,
        max_messages: int | None = None,
        fallback: str | None = None,
        fallback_model: str | None = None,
        fallback_api_key: str | None = None,
        # Caching
        cache: bool = False,
        # Tool control
        tool_choice: str | None = None,  # "auto", "required", "none"
        enabled_groups: list[str] | None = None,
        # Validation
        validator: Callable[[str], bool] | None = None,
        validation_retries: int = 0,
    ):
        self.provider, provider_config = _resolve_provider(provider, base_url)
        provider_config = provider_config.with_headers(headers)
        self.base_url = base_url
        self.headers = dict(headers or {})
        self.fallback = fallback

        self.model = model or provider_config.default_model
        if not self.model:
            raise ValueError(
                f"Provider {self.provider!r} has no default model. "
                "Pass model= to say which one to use."
            )

        self._all_tools = {t.name: t for t in (tools or [])}
        self.enabled_groups = enabled_groups
        self.api_key = api_key or _key_from_env(provider_config)
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.debug = debug

        # System prompt
        if system:
            self.system = system
        elif template and template in TEMPLATES:
            self.system = self._build_prompt(TEMPLATES[template])
        else:
            self.system = self._default_system()

        # Usage tracking
        self.usage = Usage(model=self.model, provider=provider)

        # Hooks
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.on_thinking = on_thinking
        self.on_plan = on_plan

        # Reliability
        self.retries = retries
        self.timeout = timeout
        self.max_messages = max_messages

        # Caching
        self.cache = cache

        # Tool control
        self.tool_choice = tool_choice

        # Validation
        self.validator = validator
        self.validation_retries = validation_retries

        # Memory setup
        self.session = session
        self.memory = memory or (JSONMemory() if session else None)

        if not self.api_key:
            raise ValueError(
                f"API key required. Set {provider_config.env_var} or pass api_key."
            )

        # Create client
        self.client = self._build_client(provider_config, self.api_key)

        # The fallback is a different provider, so it needs its own key and its
        # own model. Sending self.model to it would name a model it does not have.
        self.fallback_client = None
        self.fallback_model = None
        if fallback:
            if fallback not in PROVIDERS:
                raise ValueError(
                    f"Unknown fallback provider: {fallback}. Use: {list(PROVIDERS)}"
                )
            fallback_config = PROVIDERS[fallback]
            fallback_key = fallback_api_key or _key_from_env(fallback_config)
            if not fallback_key:
                raise ValueError(
                    f"Fallback provider '{fallback}' needs an API key. "
                    f"Set {fallback_config.env_var} or pass fallback_api_key."
                )
            self.fallback_model = fallback_model or fallback_config.default_model
            if not self.fallback_model:
                raise ValueError(
                    f"Fallback provider {fallback!r} has no default model. "
                    "Pass fallback_model= to say which one to use."
                )
            self.fallback_client = self._build_client(fallback_config, fallback_key)

        # Load existing session
        self.messages: list[Message] = []
        if self.session and self.memory:
            loaded = self.memory.load(self.session)
            if loaded:
                self.messages = loaded

    def _build_client(
        self, config: Provider, api_key: str
    ) -> LLMClient | AnthropicClient:
        # Without this the client kept its own 60s ceiling and Agent(timeout=300)
        # still died at 60.
        timeout = self.timeout if self.timeout else DEFAULT_REQUEST_TIMEOUT
        if config.dialect == "anthropic":
            client = AnthropicClient(
                api_key=api_key,
                base_url=config.base_url,
                timeout=timeout,
                extra_headers=dict(config.headers),
            )
            if self.max_tokens:
                client.max_tokens = self.max_tokens
            return client
        return LLMClient(
            api_key=api_key,
            base_url=config.base_url,
            timeout=timeout,
            max_tokens=self.max_tokens,
            extra_headers=dict(config.headers),
        )

    async def aclose(self) -> None:
        """Release pooled HTTP connections."""
        await self.client.aclose()
        if self.fallback_client is not None:
            await self.fallback_client.aclose()

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    @property
    def tools(self) -> dict[str, Tool]:
        if not self.enabled_groups:
            return self._all_tools
        return {
            name: t
            for name, t in self._all_tools.items()
            if t.group is None or t.group in self.enabled_groups
        }

    def enable_group(self, group: str) -> None:
        if self.enabled_groups is None:
            self.enabled_groups = []
        if group not in self.enabled_groups:
            self.enabled_groups.append(group)

    def disable_group(self, group: str) -> None:
        if self.enabled_groups and group in self.enabled_groups:
            self.enabled_groups.remove(group)

    def _default_system(self) -> str:
        tool_names = ", ".join(self.tools.keys()) if self.tools else "none"
        return (
            f"You are a helpful assistant with access to tools: {tool_names}. "
            "When you need to use a tool, call it. "
            "When you have the final answer, respond directly. Be concise."
        )

    def _build_prompt(self, base: str) -> str:
        if self.tools:
            tool_names = ", ".join(self.tools.keys())
            return f"{base} You have access to tools: {tool_names}."
        return base

    def save(self, path: str) -> None:
        data = [m.to_dict() for m in self.messages]
        write_json_atomically(Path(path), data)

    def load(self, path: str) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.messages = [Message.from_dict(m) for m in data]

    def clear(self) -> None:
        self.messages = []
        if self.session and self.memory:
            self.memory.delete(self.session)

    def _auto_save(self) -> None:
        if self.session and self.memory:
            self.memory.save(self.session, self.messages)

    def _trim_messages(self) -> None:
        """Drop old turns, but never split a tool call from its result.

        Slicing purely by count can leave a `tool` message whose assistant
        `tool_calls` message was cut, which both OpenAI-compatible APIs and
        Anthropic reject with a 400. So the window is snapped back to the
        nearest user turn, even when that keeps a couple of messages more than
        max_messages asked for.
        """
        if not self.max_messages or len(self.messages) <= self.max_messages:
            return

        system = None
        body = self.messages
        if body[0].role == "system":
            system = body[0]
            body = body[1:]

        budget = self.max_messages - (1 if system else 0)
        if budget < 1:
            self.messages = [system] if system else body[-1:]
            return

        start = max(0, len(body) - budget)
        while start > 0 and body[start].role != "user":
            start -= 1
        if body[start].role != "user":
            start = 0  # No user turn in range: keep everything rather than break it.

        self.messages = ([system] if system else []) + body[start:]

    async def _chat_with_retry(
        self,
        model: str,
        messages: list[Message],
        tools: list[Tool] | None,
        images: list[tuple[str, str]] | None = None,
    ) -> Message:
        last_error: Exception | None = None

        # Each attempt pairs a client with the model that client actually serves.
        attempts: list[tuple[Any, str]] = [(self.client, model)]
        if self.fallback_client and self.fallback_model:
            attempts.append((self.fallback_client, self.fallback_model))

        for index, (client, client_model) in enumerate(attempts):
            if index > 0 and self.debug:
                print(f"[Fallback] Switching to {self.fallback} ({client_model})")

            for attempt in range(self.retries + 1):
                try:
                    if self.timeout:
                        result = await asyncio.wait_for(
                            client.chat(
                                model=client_model,
                                messages=messages,
                                tools=tools,
                                tool_choice=self.tool_choice,
                                images=images,
                            ),
                            timeout=self.timeout,
                        )
                    else:
                        result = await client.chat(
                            model=client_model,
                            messages=messages,
                            tools=tools,
                            tool_choice=self.tool_choice,
                            images=images,
                        )
                    self.usage.add(
                        client.last_input_tokens,
                        client.last_output_tokens,
                    )
                    return result
                except Exception as e:
                    last_error = e
                    if not _is_retryable(e):
                        if self.debug:
                            print(f"[Error] {e}. Not retryable.")
                        break
                    if attempt < self.retries:
                        wait = _retry_delay(e, attempt)
                        if self.debug:
                            print(f"[Retry {attempt + 1}] {e}. Waiting {wait:.1f}s...")
                        await asyncio.sleep(wait)

        raise last_error or RuntimeError("Request failed")

    async def _run_with_plan(
        self,
        prompt: str,
        output: type[T] | None = None,
        images: list[str] | None = None,
    ) -> str | T:
        """Run with planning: create plan first, then execute step by step."""
        # Create plan
        plan_prompt = (
            f"Create a step-by-step plan to accomplish this task:\n\n{prompt}\n\n"
            "Respond with a numbered list of steps. Be specific and actionable."
        )

        if not self.messages:
            self.messages = [Message(role="system", content=self.system)]

        self.messages.append(Message(role="user", content=plan_prompt))

        plan_response = await self._chat_with_retry(
            model=self.model,
            messages=self.messages,
            tools=None,  # No tools for planning
        )
        self.messages.append(plan_response)
        plan_text = plan_response.content

        if self.debug:
            print(f"[Plan]\n{plan_text}\n")

        if self.on_plan:
            self.on_plan(plan_text)

        # Execute the plan
        execute_prompt = (
            f"Now execute this plan step by step. Use your tools as needed.\n\n"
            f"Plan:\n{plan_text}\n\n"
            f"Original task: {prompt}"
        )

        # Run normally from here
        return await self.run(execute_prompt, output=output, images=images, plan=False)

    async def _execute_tools_parallel(
        self, tool_calls: list[dict]
    ) -> list[tuple[str, str, str]]:
        async def execute_one(tc: dict, index: int) -> tuple[str, str, str]:
            function = tc.get("function") or {}
            tool_name = function.get("name") or "unknown"
            # Some providers omit the id; the API still needs one to match on.
            tool_id = tc.get("id") or f"call_{index}"

            # Everything below reports failures as tool results. A malformed
            # argument string or a broken hook must not abort the whole run.
            try:
                raw_args = function.get("arguments") or "{}"
                tool_args = (
                    json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                )
                if not isinstance(tool_args, dict):
                    raise TypeError("tool arguments must be a JSON object")
            except Exception as e:
                result = f"Error: could not parse arguments for '{tool_name}': {e}"
                self._report_tool_result(tool_name, result)
                return tool_name, tool_id, result

            if self.debug:
                print(f"[Call] {tool_name}({tool_args})")

            if self.on_tool_call:
                try:
                    self.on_tool_call(tool_name, tool_args)
                except Exception as e:
                    if self.debug:
                        print(f"[Hook] on_tool_call raised: {e}")

            if tool_name not in self.tools:
                result = f"Error: Unknown tool '{tool_name}'"
            else:
                try:
                    result = await self.tools[tool_name].call(**tool_args)
                except asyncio.TimeoutError:
                    result = f"Error: Tool '{tool_name}' timed out"
                except TypeError as e:
                    result = f"Error: bad arguments for '{tool_name}': {e}"
                except Exception as e:
                    result = f"Error: {e}"

            self._report_tool_result(tool_name, result)
            return tool_name, tool_id, result

        results = await asyncio.gather(
            *[execute_one(tc, i) for i, tc in enumerate(tool_calls)],
            return_exceptions=True,
        )

        # gather with return_exceptions can only surface a bug in execute_one
        # itself; turn it into a tool result rather than losing the whole turn.
        resolved: list[tuple[str, str, str]] = []
        for index, item in enumerate(results):
            if isinstance(item, BaseException):
                call = tool_calls[index]
                name = (call.get("function") or {}).get("name") or "unknown"
                call_id = call.get("id") or f"call_{index}"
                resolved.append((name, call_id, f"Error: {item}"))
            else:
                resolved.append(item)
        return resolved

    def _report_tool_result(self, tool_name: str, result: str) -> None:
        if self.debug:
            print(f"[Result] {result}")
        if self.on_tool_result:
            try:
                self.on_tool_result(tool_name, result)
            except Exception as e:
                if self.debug:
                    print(f"[Hook] on_tool_result raised: {e}")

    async def run(
        self,
        prompt: str,
        output: type[T] | None = None,
        images: list[str] | None = None,
        plan: bool = False,
    ) -> str | T:
        """Run the agent. Pass output=SomeDataclass for structured output."""
        if plan:
            return await self._run_with_plan(prompt, output, images)

        if not self.messages:
            self.messages = [Message(role="system", content=self.system)]

        structured = bool(output and is_dataclass(output))
        user_prompt = prompt
        if structured:
            schema = _schema_from_dataclass(output)
            user_prompt = (
                f"{prompt}\n\n"
                f"Respond with JSON matching this schema:\n"
                f"```json\n{json.dumps(schema, indent=2)}\n```"
            )

        user_message = Message(role="user", content=user_prompt)

        # The key is computed after the system prompt is in place, so two agents
        # with different personalities no longer share an entry.
        cache_key = None
        if self.cache and not images:
            cache_key = _cache_key(
                self.messages + [user_message],
                self.model,
                self.provider,
                list(self.tools.values()),
                self.tool_choice,
            )
            cached = _response_cache.get(cache_key)
            if cached is not None:
                if self.debug:
                    print("[Cache] Hit")
                # A hit still has to advance the conversation, or the next turn
                # is built on a history missing this exchange.
                self.messages.append(user_message)
                self.messages.append(Message(role="assistant", content=cached))
                self._trim_messages()
                self._auto_save()
                if structured:
                    return _parse_structured(cached, output)
                return cached

        self.messages.append(user_message)
        self._trim_messages()

        loaded_images = None
        if images:
            loaded_images = [_load_image(path) for path in images]

        for validation_attempt in range(self.validation_retries + 1):
            for step in range(self.max_steps):
                if self.debug:
                    print(f"\n[Step {step + 1}/{self.max_steps}]")

                response = await self._chat_with_retry(
                    model=self.model,
                    messages=self.messages,
                    tools=list(self.tools.values()) if self.tools else None,
                    images=loaded_images if step == 0 else None,
                )
                self.messages.append(response)

                if response.content and self.on_thinking:
                    self.on_thinking(response.content)

                if not response.tool_calls:
                    if self.debug:
                        print(f"[Final] {response.content}")

                    validated = True
                    if self.validator and not self.validator(response.content):
                        validated = False
                        if validation_attempt < self.validation_retries:
                            if self.debug:
                                print("[Validation] Failed, retrying...")
                            self.messages.append(
                                Message(role="user", content=VALIDATION_RETRY_PROMPT)
                            )
                            break  # Break inner loop, continue validation loop
                        if self.debug:
                            print("[Validation] Failed, no more retries")

                    self._auto_save()

                    if cache_key and validated:
                        _response_cache.set(cache_key, response.content)

                    if structured:
                        return _parse_structured(response.content, output)
                    return response.content

                results = await self._execute_tools_parallel(response.tool_calls)

                for tool_name, tool_id, result in results:
                    self.messages.append(
                        Message(
                            role="tool",
                            content=result,
                            tool_call_id=tool_id,
                            name=tool_name,
                        )
                    )

                # A long tool loop grows the history too; trimming only on the
                # way in let a single run blow past max_messages.
                self._trim_messages()
            else:
                break

        self._auto_save()
        # Returning this as a plain string made a failure indistinguishable
        # from an answer, and broke the return type when output= was set.
        last_assistant = next(
            (m.content for m in reversed(self.messages) if m.role == "assistant"), ""
        )
        raise MaxStepsError(self.max_steps, partial=last_assistant)

    def run_sync(
        self,
        prompt: str,
        output: type[T] | None = None,
        images: list[str] | None = None,
        plan: bool = False,
    ) -> str | T:
        return asyncio.run(self.run(prompt, output=output, images=images, plan=plan))

    def _clone(self) -> Agent:
        """A fresh agent with this one's configuration but no history."""
        clone = Agent(
            model=self.model,
            tools=list(self._all_tools.values()) if self._all_tools else None,
            api_key=self.api_key,
            max_steps=self.max_steps,
            max_tokens=self.max_tokens,
            system=self.system,
            debug=self.debug,
            provider=None if self.base_url else self.provider,
            base_url=self.base_url,
            headers=dict(self.headers) if self.headers else None,
            on_tool_call=self.on_tool_call,
            on_tool_result=self.on_tool_result,
            on_thinking=self.on_thinking,
            on_plan=self.on_plan,
            retries=self.retries,
            timeout=self.timeout,
            max_messages=self.max_messages,
            fallback=self.fallback,
            fallback_model=self.fallback_model,
            fallback_api_key=(
                self.fallback_client.api_key if self.fallback_client else None
            ),
            cache=self.cache,
            tool_choice=self.tool_choice,
            enabled_groups=(
                list(self.enabled_groups) if self.enabled_groups is not None else None
            ),
            validator=self.validator,
            validation_retries=self.validation_retries,
        )
        # Reuse this agent's pooled connections rather than opening new ones.
        clone.client = self.client
        clone.fallback_client = self.fallback_client
        return clone

    async def batch(
        self,
        prompts: list[str],
        max_concurrency: int = DEFAULT_BATCH_CONCURRENCY,
    ) -> list[str]:
        """Answer many prompts independently, at most max_concurrency at a time."""
        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async def run_one(prompt: str) -> str:
            async with semaphore:
                fresh = self._clone()
                result = await fresh.run(prompt)
                # Carry the real request count over, not one per prompt: a
                # prompt that used three tool steps made four API calls.
                self.usage.input_tokens += fresh.usage.input_tokens
                self.usage.output_tokens += fresh.usage.output_tokens
                self.usage.total_tokens += fresh.usage.total_tokens
                self.usage.requests += fresh.usage.requests
                return result

        return await asyncio.gather(*[run_one(p) for p in prompts])

    async def _stream_with_retry(
        self,
        model: str,
        messages: list[Message],
        tools: list[Tool] | None,
        images: list[tuple[str, str]] | None = None,
    ) -> AsyncIterator[tuple[str, Message | None]]:
        last_error: Exception | None = None

        attempts: list[tuple[Any, str]] = [(self.client, model)]
        if self.fallback_client and self.fallback_model:
            attempts.append((self.fallback_client, self.fallback_model))

        for index, (client, client_model) in enumerate(attempts):
            if index > 0 and self.debug:
                print(f"[Fallback] Switching to {self.fallback} ({client_model})")

            for attempt in range(self.retries + 1):
                produced = False
                try:
                    async for chunk, msg in client.chat_stream(
                        model=client_model,
                        messages=messages,
                        tools=tools,
                        tool_choice=self.tool_choice,
                        images=images,
                    ):
                        produced = True
                        yield chunk, msg
                    self.usage.add(client.last_input_tokens, client.last_output_tokens)
                    return
                except Exception as e:
                    last_error = e
                    # Once bytes have reached the caller there is no clean way
                    # to start over, so only pre-first-chunk failures retry.
                    if produced or not _is_retryable(e):
                        raise
                    if attempt < self.retries:
                        wait = _retry_delay(e, attempt)
                        if self.debug:
                            print(f"[Retry {attempt + 1}] {e}. Waiting {wait:.1f}s...")
                        await asyncio.sleep(wait)

        raise last_error or RuntimeError("Request failed")

    async def stream(
        self,
        prompt: str,
        images: list[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a run as typed events: text, tool_call, tool_result, done."""
        if not self.messages:
            self.messages = [Message(role="system", content=self.system)]
        self.messages.append(Message(role="user", content=prompt))
        self._trim_messages()

        loaded_images = [_load_image(path) for path in images] if images else None

        for step in range(self.max_steps):
            if self.debug:
                print(f"\n[Step {step + 1}/{self.max_steps}]")

            response: Message | None = None
            async for chunk, msg in self._stream_with_retry(
                model=self.model,
                messages=self.messages,
                tools=list(self.tools.values()) if self.tools else None,
                images=loaded_images if step == 0 else None,
            ):
                if chunk:
                    yield StreamEvent(type="text", content=chunk)
                if msg:
                    response = msg

            if response is None:
                return

            self.messages.append(response)

            if response.content and self.on_thinking:
                self.on_thinking(response.content)

            if not response.tool_calls:
                if self.debug:
                    print("\n[Final]")
                self._auto_save()
                yield StreamEvent(type="done", content=response.content)
                return

            for index, call in enumerate(response.tool_calls):
                function = call.get("function") or {}
                yield StreamEvent(
                    type="tool_call",
                    name=function.get("name") or "unknown",
                    id=call.get("id") or f"call_{index}",
                    arguments=_safe_arguments(function.get("arguments")),
                )

            results = await self._execute_tools_parallel(response.tool_calls)

            for tool_name, tool_id, result in results:
                self.messages.append(
                    Message(
                        role="tool",
                        content=result,
                        tool_call_id=tool_id,
                        name=tool_name,
                    )
                )
                yield StreamEvent(
                    type="tool_result",
                    name=tool_name,
                    id=tool_id,
                    content=result,
                )

            self._trim_messages()

        self._auto_save()
        yield StreamEvent(type="done", content=MAX_STEPS_MESSAGE)


def clear_cache() -> None:
    """Drop every cached response."""
    _response_cache.clear()


def set_cache_size(maxsize: int) -> None:
    """Change how many responses the shared cache keeps (default 128)."""
    _response_cache.resize(maxsize)


async def completion(
    prompt: str,
    model: str | None = None,
    tools: list[Tool] | None = None,
    api_key: str | None = None,
    provider: str = "mistral",
) -> str:
    agent = Agent(model=model, tools=tools, api_key=api_key, provider=provider)
    return await agent.run(prompt)


def completion_sync(
    prompt: str,
    model: str | None = None,
    tools: list[Tool] | None = None,
    api_key: str | None = None,
    provider: str = "mistral",
) -> str:
    return asyncio.run(completion(prompt, model, tools, api_key, provider))


END = "__end__"


class Graph:
    """
    Simple graph for multi-agent workflows.

    Example:
        graph = Graph()
        graph.add_node("research", researcher)
        graph.add_node("write", writer)
        graph.add_edge("research", "write")
        graph.set_entry("research")
        result = await graph.run("Write about AI")
    """

    def __init__(self):
        self.nodes: dict[str, Agent | Callable] = {}
        self.edges: dict[str, str] = {}
        self.conditional_edges: dict[str, Callable[[dict], str]] = {}
        self.entry: str | None = None

    def add_node(self, name: str, node: Agent | Callable[[dict], dict]) -> None:
        """Add a node. Can be an Agent or a function(state) -> state.

        The first node added becomes the entry point. Call set_entry() to
        choose a different one.
        """
        self.nodes[name] = node
        if self.entry is None:
            self.entry = name

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add a direct edge from one node to another."""
        self.edges[from_node] = to_node

    def add_conditional_edge(
        self, from_node: str, condition: Callable[[dict], str]
    ) -> None:
        """Add conditional edge. condition(state) returns next node name or END."""
        self.conditional_edges[from_node] = condition

    def set_entry(self, name: str) -> None:
        """Set the entry point node."""
        self.entry = name

    async def run(self, prompt: str, state: dict | None = None) -> dict:
        """Run the graph and return the final state.

        Agent nodes run on a per-run copy, so two runs of the same graph do not
        inherit each other's conversation history.
        """
        if not self.entry:
            raise ValueError("Graph has no nodes. Add one with graph.add_node().")

        current_state = state or {}
        current_state["input"] = prompt
        current_state["output"] = None

        agents = {
            name: node._clone()
            for name, node in self.nodes.items()
            if isinstance(node, Agent)
        }

        current_node = self.entry
        visited: list[str] = []

        while current_node != END:
            if current_node not in self.nodes:
                raise ValueError(
                    f"Unknown node: {current_node}. Known nodes: {list(self.nodes)}"
                )

            if len(visited) > 100:
                raise RuntimeError("Graph exceeded 100 steps. Possible infinite loop.")

            visited.append(current_node)

            if current_node in agents:
                input_text = current_state.get("output") or current_state["input"]
                result = await agents[current_node].run(input_text)
                current_state["output"] = result
                current_state[current_node] = result
            else:
                returned = self.nodes[current_node](current_state)
                if not isinstance(returned, dict):
                    raise TypeError(
                        f"Node '{current_node}' returned {type(returned).__name__}; "
                        "a function node must return the state dict."
                    )
                current_state = returned

            if current_node in self.conditional_edges:
                current_node = self.conditional_edges[current_node](current_state)
            elif current_node in self.edges:
                current_node = self.edges[current_node]
            else:
                current_node = END

        current_state["visited"] = visited
        return current_state

    def run_sync(self, prompt: str, state: dict | None = None) -> dict:
        """Sync version of run()."""
        return asyncio.run(self.run(prompt, state))


async def chain(agents: list[Agent], prompt: str) -> str:
    """
    Run agents in sequence. Output of each agent feeds into the next.

    Example:
        researcher = Agent(template="researcher")
        writer = Agent(template="creative")
        result = await chain([researcher, writer], "Explain quantum computing")
    """
    result = prompt
    for agent in agents:
        result = await agent.run(result)
    return result


def chain_sync(agents: list[Agent], prompt: str) -> str:
    """Sync version of chain()."""
    return asyncio.run(chain(agents, prompt))


class Router:
    """
    Route prompts to different agents based on a function.

    Example:
        def route(prompt: str) -> str:
            if "code" in prompt.lower():
                return "coder"
            return "default"

        router = Router(
            agents={"coder": coder_agent, "default": default_agent},
            route=route,
        )
        result = await router.run("Write a Python function")
    """

    def __init__(
        self,
        agents: dict[str, Agent],
        route: Callable[[str], str] | None = None,
        model: str | None = None,
        provider: str = "mistral",
        api_key: str | None = None,
    ):
        self.agents = agents
        self._route_fn = route
        self._llm_router = None

        if route is None:
            # Use LLM-based routing
            agent_list = "\n".join(f"- {name}" for name in agents.keys())
            system = (
                "You are a router. Given a user query, respond with ONLY the name "
                f"of the most appropriate agent. Available agents:\n{agent_list}"
            )
            self._llm_router = Agent(
                model=model,
                provider=provider,
                api_key=api_key,
                system=system,
                max_steps=1,
            )

    async def route(self, prompt: str) -> str:
        """Determine which agent to use."""
        if self._route_fn:
            return self._route_fn(prompt)

        # Routing is stateless: without this the router accumulated every
        # prompt it had ever seen and its choices drifted.
        self._llm_router.clear()
        result = await self._llm_router.run(prompt)
        agent_name = result.strip().lower()

        # Find matching agent
        for name in self.agents:
            if name.lower() == agent_name:
                return name

        # Fallback to first agent
        return next(iter(self.agents))

    async def run(self, prompt: str, plan: bool = False) -> str:
        """Route and run the appropriate agent."""
        agent_name = await self.route(prompt)
        return await self.agents[agent_name].run(prompt, plan=plan)

    def run_sync(self, prompt: str, plan: bool = False) -> str:
        """Sync version of run()."""
        return asyncio.run(self.run(prompt, plan=plan))
