<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/static/img/banner-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/static/img/banner-light.svg">
    <img src="docs/static/img/banner-light.svg" alt="pureagents" height="60">
  </picture>
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

A Python library for building LLM agents. ~1,500 lines you can read in an afternoon. Tools, streaming, memory, structured outputs, chaining, routing, planning, graphs. No magic, no abstractions you don't need.

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
| Hooks | `on_tool_call=fn`, `on_tool_result=fn`, `on_thinking=fn` |
| Images | `images=["photo.jpg"]` for vision models |
| Tool groups | `@tool(group="web")`, `agent.enable_group("web")` |
| Batch | `agent.batch([...])` |
| Chaining | `chain([agent1, agent2], prompt)` |
| Routing | `Router(agents={...}, route=fn)` |
| Planning | `agent.run(prompt, plan=True)` |
| Graph | `Graph()` with nodes and conditional edges |
| Retry | `retries=3` |
| Timeout | `timeout=30.0` (agent), `@tool(timeout=10)` (per-tool) |
| Context limit | `max_messages=20` |
| Usage tracking | `agent.usage` (tokens and cost) |

All features are optional. One parameter enables one feature.

## Providers

```python
# Mistral (default)
agent = Agent(provider="mistral", model="mistral-large-latest")

# OpenAI
agent = Agent(provider="openai", model="gpt-5.2-instant")

# Anthropic
agent = Agent(provider="anthropic", model="claude-opus-5")
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


result = await agent.run("Analyse this review: 'Great product!'", output=Analysis)
```

## Chaining

Sequential agent pipelines:

```python
from pure_agents import Agent, chain

researcher = Agent(system="You research topics thoroughly.")
writer = Agent(system="You write clear, concise summaries.")

result = await chain([researcher, writer], "Explain quantum computing")
```

## Routing

Dynamic agent selection:

```python
from pure_agents import Agent, Router


async def classify(prompt: str) -> str:
    return "technical" if "code" in prompt.lower() else "general"


router = Router(
    agents={
        "technical": Agent(system="You are a coding expert."),
        "general": Agent(system="You are a helpful assistant."),
    },
    route=classify,
)

result = await router.run("Write a Python function")
```

## Planning

Think before acting:

```python
agent = Agent(tools=[search, calculate])

# Agent creates a plan, then executes it
result = await agent.run(
    "Find the population of Spain and calculate 10% of it", plan=True
)
```

## Graph

Multi-agent workflows with conditional edges:

```python
from pure_agents import Agent, Graph, END

graph = Graph()
graph.add_node("research", Agent(system="Research the topic."))
graph.add_node("write", Agent(system="Write the content."))
graph.add_node("review", Agent(system="Review and improve."))

graph.add_edge("research", "write")
graph.add_conditional_edge(
    "write", lambda state: "review" if state.get("needs_review") else END
)
graph.add_edge("review", END)

result = await graph.run("Write about AI agents")
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
