---
sidebar_position: 4
---

# Streaming

Get responses in real-time, token by token.

## Basic streaming

```python
from pure_agents import Agent

agent = Agent()

async for event in agent.stream("Explain quantum computing"):
    if event.type == "text":
        print(event.content, end="", flush=True)
```

Output appears as it's generated, not all at once.

## Events

`stream()` yields `StreamEvent` objects. Each has a `type`:

| `type` | When | Fields you'll use |
|--------|------|-------------------|
| `text` | A chunk of the answer arrived | `content` |
| `tool_call` | The model asked for a tool | `name`, `arguments`, `id` |
| `tool_result` | That tool finished | `name`, `content`, `id` |
| `done` | The run finished | `content` (the full answer) |

## With tools

Tool calls are visible as they happen:

```python
@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny, 22°C in {city}"


agent = Agent(tools=[get_weather])

async for event in agent.stream("What's the weather in Madrid?"):
    if event.type == "text":
        print(event.content, end="", flush=True)
    elif event.type == "tool_call":
        print(f"\n[{event.name}({event.arguments})]")
    elif event.type == "tool_result":
        print(f"[-> {event.content}]")
```

Output:

```
[get_weather({'city': 'Madrid'})]
[-> Sunny, 22°C in Madrid]
The weather in Madrid is sunny with a temperature of 22°C.
```

## Collecting the full response

The `done` event carries the complete answer:

```python
async for event in agent.stream("Hello"):
    if event.type == "done":
        full_response = event.content
```

## Images

```python
async for event in agent.stream("What's in this photo?", images=["photo.jpg"]):
    ...
```

## Reliability

Streaming shares `run()`'s reliability features: retries, timeout, provider
fallback, and token accounting. Retries only apply before the first chunk
reaches you, since a stream already in flight cannot be restarted cleanly.

```python
agent = Agent(retries=3, timeout=120.0, fallback="anthropic")

async for event in agent.stream("Write a long essay"):
    ...

print(agent.usage.total_tokens)
```

## Provider support

Streaming works with all providers:

```python
# Mistral
agent = Agent()

# OpenAI
agent = Agent(provider="openai")

# Anthropic
agent = Agent(provider="anthropic")
```
