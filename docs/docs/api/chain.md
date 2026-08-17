---
sidebar_position: 8
---

# chain

Run agents in sequence.

## Function

```python
async def chain(agents: list[Agent], prompt: str) -> str
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `agents` | `list[Agent]` | Agents to run in order |
| `prompt` | `str` | Initial prompt |

**Returns:** Final output from last agent.

## chain_sync

```python
def chain_sync(agents: list[Agent], prompt: str) -> str
```

Synchronous version.

## How it works

```
prompt → agent1.run() → output → agent2.run() → output → agent3.run() → result
```

Each agent receives the previous agent's output as its prompt.

## Example

```python
from pure_agents import Agent, chain

researcher = Agent(template="researcher")
writer = Agent(template="creative")

result = await chain([researcher, writer], "The history of Python")
```

## Sync example

```python
from pure_agents import chain_sync

result = chain_sync([agent1, agent2], "My prompt")
```
