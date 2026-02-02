---
sidebar_position: 1
slug: /
---

# pureagents

LLM agents without the complexity. Powerful, but simple.

## Philosophy

Most agent frameworks are over-engineered. Thousands of lines, dozens of abstractions, patterns on patterns.

**pureagents is different:**

- **Powerful**: All the features you need, from basics to production
- **Simple**: ~1,500 lines. Read the entire codebase in 30 minutes
- **Elegant**: Clean API. One decorator, one class. No magic
- **Buildable**: Fork it, modify it, build on top. Not a black box

Many features ≠ complicated. Every feature is optional and modular.

## Features

### Core
| Feature | Usage |
|---------|-------|
| **Tools** | `@tool` decorator |
| **Providers** | `provider="openai"` |
| **Streaming** | `agent.stream()` |
| **Memory** | `session="my-chat"` |
| **Structured outputs** | `output=MyDataclass` |
| **Hooks** | `on_tool_call=fn` |
| **Templates** | `template="coder"` |
| **Batch** | `agent.batch([...])` |

### Reliability
| Feature | Usage |
|---------|-------|
| **Retry** | `retries=3` |
| **Timeout** | `timeout=30.0` |
| **Context limit** | `max_messages=20` |
| **Fallback** | `fallback="mistral"` |
| **Caching** | `cache=True` |

### Tools
| Feature | Usage |
|---------|-------|
| **Tool timeout** | `@tool(timeout=10)` |
| **Tool groups** | `@tool(group="web")` |
| **Force tool use** | `tool_choice="required"` |
| **Parallel execution** | Automatic |

### Orchestration
| Feature | Usage |
|---------|-------|
| **Chaining** | `chain([agent1, agent2], prompt)` |
| **Routing** | `Router(agents={...})` |
| **Planning** | `plan=True` |
| **Graph** | `Graph()` with conditional edges |

### Advanced
| Feature | Usage |
|---------|-------|
| **Images** | `images=["photo.jpg"]` |
| **Validation** | `validator=fn` |
| **Usage tracking** | `agent.usage` |

## Quick example

```python
from pure_agents import Agent, tool

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for {query}..."

agent = Agent(tools=[search])
result = await agent.run("Find the weather in Madrid")
```

## Installation

```bash
pip install pureagents
```

```bash
export MISTRAL_API_KEY="your-key"  # Default provider
```
