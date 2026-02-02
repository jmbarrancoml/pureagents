"""Tool decorator and Tool class."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints


@dataclass
class Tool:
    """A tool that an agent can use."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    timeout: float | None = None
    group: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.parameters.keys()),
                },
            },
        }

    async def call(self, **kwargs: Any) -> str:
        async def _execute() -> str:
            result = self.fn(**kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return str(result)

        if self.timeout:
            return await asyncio.wait_for(_execute(), timeout=self.timeout)
        return await _execute()


def _python_type_to_json_schema(py_type: type) -> dict[str, str]:
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }
    return type_map.get(py_type, {"type": "string"})


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

        parameters = {}
        for param_name, param in sig.parameters.items():
            if param_name == "return":
                continue
            param_type = hints.get(param_name, str)
            parameters[param_name] = _python_type_to_json_schema(param_type)

        return Tool(
            name=name,
            description=description,
            parameters=parameters,
            fn=f,
            timeout=timeout,
            group=group,
        )

    # Support both @tool and @tool(timeout=10)
    if fn is not None:
        return decorator(fn)
    return decorator
