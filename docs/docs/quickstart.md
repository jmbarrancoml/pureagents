---
sidebar_position: 2
---

# Quick start

Get up and running in 5 minutes.

## Installation

```bash
pip install pureagents
```

```bash
export MISTRAL_API_KEY=your-key
```

## Your first agent

```python
from pure_agents import Agent, tool

@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

agent = Agent(tools=[greet])
result = await agent.run("Say hello to Maria")
```

## Common patterns

### Streaming

```python
async for chunk in agent.stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

### Structured outputs

```python
from dataclasses import dataclass

@dataclass
class Analysis:
    sentiment: str
    confidence: float

result = await agent.run("Analyse: I love this!", output=Analysis)
```

### Memory

```python
agent = Agent(session="my-chat")
```

### Different providers

```python
agent = Agent(provider="openai")
agent = Agent(provider="anthropic")
agent = Agent(provider="openai", fallback="mistral")  # Failover
```

### Templates

```python
agent = Agent(template="coder")
agent = Agent(template="researcher")
```

### Reliability

```python
agent = Agent(
    retries=3,
    timeout=30.0,
    max_messages=20,
    cache=True,
)
```

### Tool control

```python
@tool(timeout=10, group="web")
def search(query: str) -> str:
    """Search with timeout."""
    return "..."

agent = Agent(
    tools=[search],
    tool_choice="required",
    enabled_groups=["web"],
)
```

### Batch processing

```python
results = await agent.batch(["Q1", "Q2", "Q3"])
```

### Images

```python
result = await agent.run(
    "Describe this image",
    images=["photo.jpg"],
)
```

### Validation

```python
def is_json(response: str) -> bool:
    try:
        json.loads(response)
        return True
    except:
        return False

agent = Agent(validator=is_json, validation_retries=2)
```

### Usage tracking

```python
await agent.run("Hello")
print(agent.usage.total_tokens)
print(agent.usage.cost("openai"))
```

## Sync usage

```python
result = agent.run_sync("Hello")
```

## Next steps

- [Tools guide](/docs/guides/tools)
- [Reliability guide](/docs/guides/reliability)
- [API reference](/docs/api/agent)
