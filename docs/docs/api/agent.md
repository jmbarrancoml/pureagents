---
sidebar_position: 1
---

# Agent

The main class for creating LLM agents.

## Constructor

```python
Agent(
    # Core
    model: str | None = None,
    tools: list[Tool] | None = None,
    api_key: str | None = None,
    max_steps: int = 10,
    system_prompt: str | None = None,
    template: str | None = None,
    debug: bool = False,
    provider: str = "mistral",
    session: str | None = None,
    memory: Memory | None = None,
    # Hooks
    on_tool_call: Callable | None = None,
    on_tool_result: Callable | None = None,
    on_thinking: Callable | None = None,
    on_plan: Callable | None = None,
    # Reliability
    retries: int = 0,
    timeout: float | None = None,
    max_messages: int | None = None,
    fallback: str | None = None,
    cache: bool = False,
    # Tool control
    tool_choice: str | None = None,
    enabled_groups: list[str] | None = None,
    # Validation
    validator: Callable[[str], bool] | None = None,
    validation_retries: int = 0,
)
```

## Parameters

### Core

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | Provider default | Model to use |
| `tools` | `list[Tool]` | `None` | Tools the agent can use |
| `api_key` | `str` | From env | API key |
| `max_steps` | `int` | `10` | Max ReAct loop iterations |
| `system_prompt` | `str` | Auto | Custom system prompt |
| `template` | `str` | `None` | Predefined template |
| `debug` | `bool` | `False` | Print debug info |
| `provider` | `str` | `"mistral"` | LLM provider |
| `session` | `str` | `None` | Session ID for persistence |
| `memory` | `Memory` | `JSONMemory` | Custom memory backend |

### Hooks

| Parameter | Type | Description |
|-----------|------|-------------|
| `on_tool_call` | `Callable[[str, dict], None]` | Before tool execution |
| `on_tool_result` | `Callable[[str, str], None]` | After tool execution |
| `on_thinking` | `Callable[[str], None]` | On intermediate text |
| `on_plan` | `Callable[[str], None]` | When plan is created (with `plan=True`) |

### Reliability

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `retries` | `int` | `0` | Retry attempts on failure |
| `timeout` | `float` | `None` | Request timeout (seconds) |
| `max_messages` | `int` | `None` | Max messages in context |
| `fallback` | `str` | `None` | Fallback provider |
| `cache` | `bool` | `False` | Enable response caching |

### Tool Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tool_choice` | `str` | `None` | `"auto"`, `"required"`, `"none"` |
| `enabled_groups` | `list[str]` | `None` | Enabled tool groups |

### Validation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `validator` | `Callable[[str], bool]` | `None` | Response validator |
| `validation_retries` | `int` | `0` | Retries on validation fail |

## Methods

### run

```python
async def run(
    self,
    prompt: str,
    output: type[T] | None = None,
    images: list[str] | None = None,
    plan: bool = False,
) -> str | T
```

Run the agent. Returns response or dataclass instance.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | Required | The prompt |
| `output` | `type` | `None` | Dataclass for structured output |
| `images` | `list[str]` | `None` | Image paths to include |
| `plan` | `bool` | `False` | Create plan before executing |

### run_sync

```python
def run_sync(prompt, output=None, images=None, plan=False) -> str | T
```

Synchronous version of `run()`.

### batch

```python
async def batch(self, prompts: list[str]) -> list[str]
```

Run multiple prompts in parallel.

### stream

```python
async def stream(self, prompt: str) -> AsyncIterator[str]
```

Stream response token by token.

### save / load / clear

```python
def save(self, path: str) -> None
def load(self, path: str) -> None
def clear(self) -> None
```

Manage conversation history.

### enable_group / disable_group

```python
def enable_group(self, group: str) -> None
def disable_group(self, group: str) -> None
```

Dynamically enable/disable tool groups.

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `messages` | `list[Message]` | Conversation history |
| `tools` | `dict[str, Tool]` | Active tools |
| `usage` | `Usage` | Token usage tracking |

## Example

```python
from pure_agents import Agent, tool


@tool(timeout=10, group="search")
def search(query: str) -> str:
    """Search the web."""
    return f"Results for {query}"


agent = Agent(
    provider="openai",
    fallback="mistral",
    template="researcher",
    tools=[search],
    retries=2,
    timeout=30.0,
    cache=True,
    tool_choice="auto",
    on_tool_call=lambda n, a: print(f"Calling {n}"),
)

result = await agent.run("Search for Python tutorials")
print(f"Tokens: {agent.usage.total_tokens}")
print(f"Cost: ${agent.usage.cost('openai'):.4f}")
```
