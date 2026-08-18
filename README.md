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

A Python library for building LLM agents. ~2,200 lines you can read in an afternoon. Tools, streaming, memory, structured outputs, chaining, routing, planning, graphs. No magic, no abstractions you don't need.

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
| Any endpoint | `base_url="http://localhost:11434/v1"`, or `register_provider(...)` |
| Streaming | `agent.stream()` yields `StreamEvent` objects |
| Memory | `session="my-chat"` |
| Structured outputs | `output=MyDataclass` |
| Hooks | `on_tool_call=fn`, `on_tool_result=fn`, `on_thinking=fn` |
| Images | `images=["photo.jpg"]` for vision models |
| Tool groups | `@tool(group="web")`, `agent.enable_group("web")` |
| Batch | `agent.batch([...], max_concurrency=5)` |
| Chaining | `chain([agent1, agent2], prompt)` |
| Routing | `Router(agents={...}, route=fn)` |
| Planning | `agent.run(prompt, plan=True)` |
| Graph | `Graph()` with nodes and conditional edges |
| Retry | `retries=3` (only on 429, 5xx and network failures) |
| Fallback | `fallback="anthropic"` switches provider and model |
| Timeout | `timeout=30.0` (agent), `@tool(timeout=10)` (per-tool) |
| Output limit | `max_tokens=8000` |
| Context limit | `max_messages=20` |
| Usage tracking | `agent.usage` (tokens and cost) |

All features are optional. One parameter enables one feature.

## Errors

Failures are exceptions, not strings that look like answers:

| Exception | When |
|-----------|------|
| `MaxStepsError` | The agent used every step without a final answer. `.partial` holds what it last said. |
| `StructuredOutputError` | The model's reply did not match your dataclass. `.raw` holds the reply. |
| `TruncatedResponseError` | The provider stopped at `max_tokens` mid tool call. |
| `httpx.HTTPStatusError` | The provider rejected the request. 4xx errors are not retried. |

## Providers

```python
# Mistral (default)
agent = Agent(provider="mistral", model="mistral-large-latest")

# OpenAI
agent = Agent(provider="openai", model="gpt-5.6-luna")

# Anthropic
agent = Agent(provider="anthropic", model="claude-sonnet-5")
```

API keys via environment variables (`MISTRAL_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) or `api_key=`.

Defaults are the cheapest current-generation tier that still drives a tool loop
reliably. Pass `model=` for anything else:

| Provider | Default | $/1M in | $/1M out | Step up to |
|----------|---------|--------:|---------:|------------|
| `mistral` | `mistral-large-latest` | 0.50 | 1.50 | `mistral-medium-latest` |
| `openai` | `gpt-5.6-luna` | 0.20 | 1.20 | `gpt-5.6-terra`, `gpt-5.6-sol` |
| `anthropic` | `claude-sonnet-5` | 2.00 | 10.00 | `claude-opus-5`, `claude-fable-5` |

Rates checked against each provider's pricing page on 2026-08-17.
`agent.usage.cost()` uses the same table.

### Anything else that speaks OpenAI

```python
# Ollama, LM Studio, vLLM, OpenRouter, Groq, Together, your own gateway
agent = Agent(base_url="http://localhost:11434/v1", model="llama3.3")

# Or name it once and reuse it
from pure_agents import register_provider

register_provider("ollama", base_url="http://localhost:11434/v1")
agent = Agent(provider="ollama", model="llama3.3")
```

No key needed when the server does not ask for one. See the
[providers guide](https://pureagents.dev/docs/guides/providers) for the common
base URLs and for the `dialect="anthropic"` option.

## Streaming

```python
async for event in agent.stream("Write a poem"):
    if event.type == "text":
        print(event.content, end="", flush=True)
    elif event.type == "tool_call":
        print(f"\nCalling {event.name} with {event.arguments}...")
    elif event.type == "tool_result":
        print(f"-> {event.content}")
    elif event.type == "done":
        print(f"\n[{len(event.content)} characters]")
```

Every event is a `StreamEvent` with `type`, `content`, `name`, `id` and
`arguments`. Streaming goes through the same retries, timeout, fallback and
token accounting as `run()`.

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

The first node added is the entry point. Call `graph.set_entry("write")` to
choose a different one. Each `graph.run()` gives its agents a fresh copy, so
runs don't inherit each other's history.

## State and concurrency

An `Agent` is stateful: `run()` appends to `agent.messages` and the next call
sends the whole conversation. That is what makes memory work, and it means two
things:

```python
# One agent, one conversation at a time. This corrupts the history:
await asyncio.gather(agent.run("a"), agent.run("b"))

# Use batch(), which gives each prompt its own copy:
await agent.batch(["a", "b"])

# Or start fresh:
agent.clear()
```

Connections are pooled per agent. Close them when you're done, or use the
context manager:

```python
async with Agent(tools=[search]) as agent:
    await agent.run("Find the weather in Madrid")
```

## Why

We follow [Anthropic's philosophy](https://www.anthropic.com/research/building-effective-agents): start simple, add complexity only when needed. Most agent frameworks do the opposite.

This library gives you a clean starting point that breaks the blank page problem. Instead of figuring out how to structure your agent code, you get a working foundation with sensible defaults. Then you adapt it to your needs.

- **~2,200 lines total.** Read the entire codebase in one sitting.
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
