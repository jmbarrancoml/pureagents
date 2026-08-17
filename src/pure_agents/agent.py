"""Agent class and convenience functions."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

from pure_agents.clients import PROVIDERS, AnthropicClient, LLMClient
from pure_agents.memory import JSONMemory, Memory
from pure_agents.message import Message
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

VALIDATION_RETRY_PROMPT = "Your response was invalid. Please try again."

_response_cache: dict[str, str] = {}


@dataclass
class Usage:
    """Token usage and cost tracking."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0

    # Approximate costs per 1M tokens (USD)
    _costs: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "mistral": (0.25, 0.25),
            "openai": (2.50, 10.00),
            "anthropic": (3.00, 15.00),
        }
    )

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
        self.requests += 1

    def cost(self, provider: str = "mistral") -> float:
        rates = self._costs.get(provider, (1.0, 1.0))
        input_cost = (self.input_tokens / 1_000_000) * rates[0]
        output_cost = (self.output_tokens / 1_000_000) * rates[1]
        return input_cost + output_cost

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.requests = 0


def _schema_from_dataclass(cls: type) -> dict[str, Any]:
    hints = get_type_hints(cls)
    properties = {}
    for name, hint in hints.items():
        if hint is str:
            properties[name] = {"type": "string"}
        elif hint is int:
            properties[name] = {"type": "integer"}
        elif hint is float:
            properties[name] = {"type": "number"}
        elif hint is bool:
            properties[name] = {"type": "boolean"}
        else:
            properties[name] = {"type": "string"}
    return {
        "type": "object",
        "properties": properties,
        "required": list(hints.keys()),
    }


def _parse_structured(content: str, output_cls: type[T]) -> T:
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])
    data = json.loads(text)
    return output_cls(**data)


def _cache_key(messages: list[Message], model: str) -> str:
    content = json.dumps([m.to_dict() for m in messages]) + model
    return hashlib.sha256(content.encode()).hexdigest()


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
        system_prompt: str | None = None,
        template: str | None = None,
        debug: bool = False,
        provider: str = "mistral",
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
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Use: {list(PROVIDERS)}")

        self.provider = provider
        self.fallback = fallback
        provider_config = PROVIDERS[provider]

        self.model = model or provider_config["default_model"]
        self._all_tools = {t.name: t for t in (tools or [])}
        self.enabled_groups = enabled_groups
        self.api_key = api_key or os.environ.get(provider_config["env_var"], "")
        self.max_steps = max_steps
        self.debug = debug

        # System prompt
        if system_prompt:
            self.system_prompt = system_prompt
        elif template and template in TEMPLATES:
            self.system_prompt = self._build_prompt(TEMPLATES[template])
        else:
            self.system_prompt = self._default_system_prompt()

        # Usage tracking
        self.usage = Usage()

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
                f"API key required. Set {provider_config['env_var']} or pass api_key."
            )

        # Create client
        self.client = self._create_client(provider, self.api_key)

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
            fallback_key = fallback_api_key or os.environ.get(
                fallback_config["env_var"], ""
            )
            if not fallback_key:
                raise ValueError(
                    f"Fallback provider '{fallback}' needs an API key. "
                    f"Set {fallback_config['env_var']} or pass fallback_api_key."
                )
            self.fallback_model = fallback_model or fallback_config["default_model"]
            self.fallback_client = self._create_client(fallback, fallback_key)

        # Load existing session
        self.messages: list[Message] = []
        if self.session and self.memory:
            loaded = self.memory.load(self.session)
            if loaded:
                self.messages = loaded

    def _create_client(
        self, provider: str, api_key: str
    ) -> LLMClient | AnthropicClient:
        config = PROVIDERS[provider]
        if config.get("client") == "anthropic":
            return AnthropicClient(api_key=api_key, base_url=config["base_url"])
        return LLMClient(api_key=api_key, base_url=config["base_url"])

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

    def _default_system_prompt(self) -> str:
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
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

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
        if not self.max_messages or len(self.messages) <= self.max_messages:
            return
        system = self.messages[0] if self.messages[0].role == "system" else None
        if system:
            keep = self.max_messages - 1
            self.messages = [system] + self.messages[-keep:]
        else:
            self.messages = self.messages[-self.max_messages :]

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
                    if attempt < self.retries:
                        wait = 2**attempt
                        if self.debug:
                            print(f"[Retry {attempt + 1}] {e}. Waiting {wait}s...")
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
            self.messages = [Message(role="system", content=self.system_prompt)]

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
        async def execute_one(tc: dict) -> tuple[str, str, str]:
            tool_name = tc["function"]["name"]
            tool_args = json.loads(tc["function"]["arguments"])
            tool_id = tc["id"]

            if self.debug:
                print(f"[Call] {tool_name}({tool_args})")

            if self.on_tool_call:
                self.on_tool_call(tool_name, tool_args)

            if tool_name not in self.tools:
                result = f"Error: Unknown tool '{tool_name}'"
            else:
                try:
                    result = await self.tools[tool_name].call(**tool_args)
                except asyncio.TimeoutError:
                    result = f"Error: Tool '{tool_name}' timed out"
                except Exception as e:
                    result = f"Error: {e}"

            if self.debug:
                print(f"[Result] {result}")

            if self.on_tool_result:
                self.on_tool_result(tool_name, result)

            return tool_name, tool_id, result

        results = await asyncio.gather(*[execute_one(tc) for tc in tool_calls])
        return results

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
        cache_key = None
        if self.cache and not images:
            cache_key = _cache_key(
                self.messages + [Message(role="user", content=prompt)],
                self.model,
            )
            if cache_key in _response_cache:
                if self.debug:
                    print("[Cache] Hit")
                return _response_cache[cache_key]

        if not self.messages:
            self.messages = [Message(role="system", content=self.system_prompt)]

        user_prompt = prompt
        if output and is_dataclass(output):
            schema = _schema_from_dataclass(output)
            user_prompt = (
                f"{prompt}\n\n"
                f"Respond with JSON matching this schema:\n"
                f"```json\n{json.dumps(schema, indent=2)}\n```"
            )

        self.messages.append(Message(role="user", content=user_prompt))
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

                    if self.validator and not self.validator(response.content):
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

                    if self.cache and cache_key:
                        _response_cache[cache_key] = response.content

                    if output and is_dataclass(output):
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
            else:
                break

        self._auto_save()
        return "Max steps reached without final answer."

    def run_sync(
        self,
        prompt: str,
        output: type[T] | None = None,
        images: list[str] | None = None,
        plan: bool = False,
    ) -> str | T:
        return asyncio.run(self.run(prompt, output=output, images=images, plan=plan))

    async def batch(self, prompts: list[str]) -> list[str]:

        async def run_one(prompt: str) -> str:
            fresh = Agent(
                model=self.model,
                tools=list(self._all_tools.values()) if self._all_tools else None,
                api_key=self.api_key,
                max_steps=self.max_steps,
                system_prompt=self.system_prompt,
                debug=self.debug,
                provider=self.provider,
                retries=self.retries,
                timeout=self.timeout,
                fallback=self.fallback,
                cache=self.cache,
                tool_choice=self.tool_choice,
            )
            result = await fresh.run(prompt)
            self.usage.add(fresh.usage.input_tokens, fresh.usage.output_tokens)
            return result

        return await asyncio.gather(*[run_one(p) for p in prompts])

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        if not self.messages:
            self.messages = [Message(role="system", content=self.system_prompt)]
        self.messages.append(Message(role="user", content=prompt))
        self._trim_messages()

        for step in range(self.max_steps):
            if self.debug:
                print(f"\n[Step {step + 1}/{self.max_steps}]")

            response: Message | None = None
            async for chunk, msg in self.client.chat_stream(
                model=self.model,
                messages=self.messages,
                tools=list(self.tools.values()) if self.tools else None,
                tool_choice=self.tool_choice,
            ):
                if chunk:
                    yield chunk
                if msg:
                    response = msg

            if response is None:
                return

            self.messages.append(response)

            if not response.tool_calls:
                if self.debug:
                    print("\n[Final]")
                self._auto_save()
                return

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


def clear_cache() -> None:
    _response_cache.clear()


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
        """Add a node. Can be an Agent or a function(state) -> state."""
        self.nodes[name] = node

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
        """Run the graph. Returns final state."""
        if not self.entry:
            raise ValueError("No entry point set. Use graph.set_entry()")

        current_state = state or {}
        current_state["input"] = prompt
        current_state["output"] = None

        current_node = self.entry
        visited = []

        while current_node != END:
            if current_node not in self.nodes:
                raise ValueError(f"Unknown node: {current_node}")

            if len(visited) > 100:
                raise RuntimeError("Graph exceeded 100 steps. Possible infinite loop.")

            visited.append(current_node)
            node = self.nodes[current_node]

            # Execute node
            if isinstance(node, Agent):
                input_text = current_state.get("output") or current_state["input"]
                result = await node.run(input_text)
                current_state["output"] = result
                current_state[current_node] = result
            else:
                # It's a function
                current_state = node(current_state)

            # Find next node
            if current_node in self.conditional_edges:
                current_node = self.conditional_edges[current_node](current_state)
            elif current_node in self.edges:
                current_node = self.edges[current_node]
            else:
                current_node = END

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
                system_prompt=system,
                max_steps=1,
            )

    async def route(self, prompt: str) -> str:
        """Determine which agent to use."""
        if self._route_fn:
            return self._route_fn(prompt)

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
