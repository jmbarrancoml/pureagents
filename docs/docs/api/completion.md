---
sidebar_position: 5
---

# Completion functions

Convenience functions for quick one-off completions.

## completion

```python
async def completion(
    prompt: str,
    model: str | None = None,
    tools: list[Tool] | None = None,
    api_key: str | None = None,
    provider: str = "mistral",
) -> str
```

One-liner async completion.

```python
from pure_agents import completion

result = await completion("What is the capital of France?")
print(result)  # "The capital of France is Paris."
```

### With tools

```python
from pure_agents import completion, tool


@tool
def get_time() -> str:
    """Get the current time."""
    from datetime import datetime

    return datetime.now().strftime("%H:%M")


result = await completion("What time is it?", tools=[get_time])
```

### With different provider

```python
result = await completion(
    "Explain quantum computing", provider="anthropic", model="claude-opus-4-5-20251101"
)
```

## completion_sync

```python
def completion_sync(
    prompt: str,
    model: str | None = None,
    tools: list[Tool] | None = None,
    api_key: str | None = None,
    provider: str = "mistral",
) -> str
```

Synchronous version.

```python
from pure_agents import completion_sync

result = completion_sync("What is 2 + 2?")
```

## When to use

Use completion functions for:
- Quick one-off queries
- Scripts without async
- Testing and prototyping

Use `Agent` for:
- Multi-turn conversations
- Session persistence
- Streaming responses
- More control
