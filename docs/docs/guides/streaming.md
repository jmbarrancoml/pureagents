---
sidebar_position: 4
---

# Streaming

Get responses in real-time, token by token.

## Basic streaming

```python
from pure_agents import Agent

agent = Agent()

async for chunk in agent.stream("Explain quantum computing"):
    print(chunk, end="", flush=True)
```

Output appears as it's generated, not all at once.

## With tools

Streaming works with tools too. The agent handles tool calls automatically:

```python
@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny, 22°C in {city}"

agent = Agent(tools=[get_weather])

async for chunk in agent.stream("What's the weather in Madrid?"):
    print(chunk, end="", flush=True)
```

The agent will:
1. Call `get_weather("Madrid")`
2. Stream the final response

## Debug mode with streaming

Enable debug to see tool calls:

```python
agent = Agent(tools=[get_weather], debug=True)

async for chunk in agent.stream("What's the weather in Madrid?"):
    print(chunk, end="", flush=True)
```

Output:
```
[Step 1/10]
[Call] get_weather({'city': 'Madrid'})
[Result] Sunny, 22°C in Madrid
[Step 2/10]
The weather in Madrid is sunny with a temperature of 22°C.
[Final]
```

## Collecting the full response

If you need the complete response:

```python
chunks = []
async for chunk in agent.stream("Hello"):
    chunks.append(chunk)

full_response = "".join(chunks)
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
