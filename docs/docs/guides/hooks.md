---
sidebar_position: 7
---

# Hooks

Monitor agent behaviour with callbacks.

## Available hooks

| Hook | Called when | Arguments |
|------|-------------|-----------|
| `on_tool_call` | Before a tool executes | `(name: str, args: dict)` |
| `on_tool_result` | After a tool returns | `(name: str, result: str)` |
| `on_thinking` | Agent produces intermediate text | `(text: str)` |

## Basic usage

```python
from pure_agents import Agent


def log_call(name: str, args: dict):
    print(f"Calling {name} with {args}")


def log_result(name: str, result: str):
    print(f"{name} returned: {result[:50]}...")


agent = Agent(
    on_tool_call=log_call,
    on_tool_result=log_result,
)
```

## Logging example

```python
from datetime import datetime


def timestamped_log(name: str, args: dict):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] CALL: {name}({args})")


agent = Agent(
    on_tool_call=timestamped_log,
    debug=False,  # Use hooks instead of built-in debug
)
```

## Metrics collection

```python
metrics = {"calls": 0, "errors": 0}


def track_call(name: str, args: dict):
    metrics["calls"] += 1


def track_result(name: str, result: str):
    if result.startswith("Error"):
        metrics["errors"] += 1


agent = Agent(
    on_tool_call=track_call,
    on_tool_result=track_result,
)

await agent.run("Do something")
print(f"Total calls: {metrics['calls']}, Errors: {metrics['errors']}")
```

## When to use

- **Logging**: Track what your agent is doing
- **Monitoring**: Collect metrics for observability
- **Debugging**: Understand agent behaviour
- **Auditing**: Record all tool invocations
