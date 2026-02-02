---
sidebar_position: 6
---

# Router

Route prompts to different agents.

## Constructor

```python
Router(
    agents: dict[str, Agent],
    route: Callable[[str], str] | None = None,
    model: str | None = None,
    provider: str = "mistral",
    api_key: str | None = None,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `agents` | `dict[str, Agent]` | Named agents to route between |
| `route` | `Callable` | Function `(prompt) -> agent_name`. If None, uses LLM |
| `model` | `str` | Model for LLM-based routing |
| `provider` | `str` | Provider for LLM-based routing |
| `api_key` | `str` | API key for LLM-based routing |

## Methods

### route

```python
async def route(prompt: str) -> str
```

Determine which agent to use. Returns agent name.

### run

```python
async def run(prompt: str, plan: bool = False) -> str
```

Route to appropriate agent and run it.

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | `str` | The prompt to process |
| `plan` | `bool` | If True, agent creates plan first |

### run_sync

```python
def run_sync(prompt: str, plan: bool = False) -> str
```

Synchronous version of `run()`.

## Example

```python
from pure_agents import Agent, Router

coder = Agent(template="coder")
writer = Agent(template="creative")

# Function-based routing
def route(prompt: str) -> str:
    if "code" in prompt.lower():
        return "coder"
    return "writer"

router = Router(
    agents={"coder": coder, "writer": writer},
    route=route,
)

result = await router.run("Write a sorting function")
```

## LLM-based routing

```python
# No route function = LLM decides
router = Router(
    agents={"coder": coder, "writer": writer},
)

result = await router.run("Explain recursion")
```
