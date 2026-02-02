<p align="center">
  <img src="docs/static/img/banner.svg" alt="pureagents" height="60">
</p>

<p align="center">
  <a href="https://pypi.org/project/pureagents/"><img src="https://img.shields.io/pypi/v/pureagents?style=flat-square&color=orange" alt="PyPI"></a>
  <a href="https://pypi.org/project/pureagents/"><img src="https://img.shields.io/pypi/pyversions/pureagents?style=flat-square" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="License"></a>
</p>

<p align="center">
  The simplest agent framework. And the most capable.
</p>

<p align="center">
  <a href="https://pureagents.dev">Documentation</a> ·
  <a href="https://pureagents.dev/docs/quickstart">Quick start</a> ·
  <a href="https://pureagents.dev/docs/api/agent">API</a>
</p>

---

A Python library for building LLM agents. One file you can read in 30 minutes. Tools, streaming, memory, structured outputs, multi-agent orchestration. No magic, no abstractions you don't need.

---

## Installation

```bash
pip install pureagents
```

## Usage

```python
from pure_agents import Agent, tool

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for {query}..."

agent = Agent(tools=[search])
result = await agent.run("Find the weather in Madrid")
```

## Features

| Feature | Usage |
|---------|-------|
| Tools | `@tool` decorator, type hints become JSON schemas |
| Providers | `provider="openai"` (Mistral, OpenAI, Anthropic) |
| Streaming | `agent.stream()` |
| Memory | `session="my-chat"` |
| Structured outputs | `output=MyDataclass` |
| Hooks | `on_tool_call=fn` |
| Batch | `agent.batch([...])` |
| Chaining | `chain([agent1, agent2], prompt)` |
| Routing | `Router(agents={...}, route=fn)` |
| Planning | `agent.run(prompt, plan=True)` |
| Graph | `Graph()` with nodes and conditional edges |
| Retry | `retries=3` |
| Timeout | `timeout=30.0` |
| Context limit | `max_messages=20` |
| Usage tracking | `agent.usage` |

All features are optional. One parameter enables one feature.

## Providers

```python
# Mistral (default)
agent = Agent(provider="mistral", model="mistral-large-latest")

# OpenAI
agent = Agent(provider="openai", model="gpt-4o")

# Anthropic
agent = Agent(provider="anthropic", model="claude-sonnet-4-20250514")
```

API keys via environment variables (`MISTRAL_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) or `api_key=`.

## Streaming

```python
async for event in agent.stream("Write a poem"):
    if event.type == "text":
        print(event.content, end="", flush=True)
    elif event.type == "tool_call":
        print(f"\nCalling {event.name}...")
```

## Memory

```python
agent = Agent(session="my-chat")
await agent.run("I'm Jose")
# Later...
await agent.run("What's my name?")  # Remembers

# Manual
agent.save("conversation.json")
agent.load("conversation.json")
```

## Structured outputs

```python
from dataclasses import dataclass

@dataclass
class Analysis:
    sentiment: str
    confidence: float
    keywords: list[str]

result = await agent.run(
    "Analyse this review: 'Great product!'",
    output=Analysis
)
```

## Why

We follow [Anthropic's philosophy](https://www.anthropic.com/research/building-effective-agents): start simple, add complexity only when needed. Most agent frameworks do the opposite.

This library gives you a clean starting point that breaks the blank page problem. Instead of figuring out how to structure your agent code, you get a working foundation with sensible defaults. Then you adapt it to your needs.

- **~1,500 lines total.** Read the entire codebase in one sitting.
- **No hidden behaviour.** What you see is what runs.
- **Modular by default.** Each feature is independent. Use what you need.
- **Built to be forked.** If it doesn't fit your use case, copy and modify it.

## Ideal for

- **Projects that need full control.** No black boxes.
- **Teams that want to understand their code.** Not just use it.
- **Prototypes that might become products.** Start simple, grow as needed.
- **Learning how agents work.** The code is the documentation.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

[Apache 2.0](LICENSE)
