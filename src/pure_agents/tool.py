"""Tool decorator and Tool class."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, get_type_hints

from pure_agents.schema import json_schema


@dataclass
class Tool:
    """A tool that an agent can use."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    timeout: float | None = None
    group: str | None = None
    # Parameters without a default. Marking everything required forced the
    # model to invent values for optional arguments.
    required: list[str] = field(default_factory=list)

    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": self.parameters,
            "required": list(self.required),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema(),
            },
        }

    async def call(self, **kwargs: Any) -> str:
        """Run the tool and return its result as text.

        Sync functions run on a worker thread. Calling them inline would block
        the event loop, which both serialises parallel tool calls and leaves
        asyncio.wait_for with no opportunity to fire.

        A timeout returns control to the agent straight away, but Python cannot
        kill the thread, so a runaway sync tool keeps running in the background.
        Under run_sync() that shows up at the end: asyncio.run waits for the
        thread pool to drain before returning, so the process can sit there
        after the agent has already moved on. Prefer an async tool that can be
        cancelled when the work might genuinely overrun.
        """

        async def _execute() -> str:
            if inspect.iscoroutinefunction(self.fn):
                result = await self.fn(**kwargs)
            else:
                result = await asyncio.to_thread(partial(self.fn, **kwargs))
                if inspect.isawaitable(result):
                    result = await result
            return str(result)

        if self.timeout:
            return await asyncio.wait_for(_execute(), timeout=self.timeout)
        return await _execute()


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    timeout: float | None = None,
    group: str | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Docstring becomes description, type hints become schema."""

    def decorator(f: Callable[..., Any]) -> Tool:
        name = f.__name__
        description = f.__doc__ or f"Tool: {name}"
        description = description.strip()

        # Extract parameters from type hints
        hints = get_type_hints(f)
        sig = inspect.signature(f)

        parameters: dict[str, Any] = {}
        required: list[str] = []
        for param_name, param in sig.parameters.items():
            if param_name == "return":
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            parameters[param_name] = json_schema(hints.get(param_name, str))
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return Tool(
            name=name,
            description=description,
            parameters=parameters,
            fn=f,
            timeout=timeout,
            group=group,
            required=required,
        )

    # Support both @tool and @tool(timeout=10)
    if fn is not None:
        return decorator(fn)
    return decorator
