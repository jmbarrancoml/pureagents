---
sidebar_position: 7
---

# Graph

Multi-agent workflows with conditional routing.

## Constructor

```python
Graph()
```

No parameters. Configure with methods.

## Methods

### add_node

```python
def add_node(name: str, node: Agent | Callable[[dict], dict]) -> None
```

Add a node to the graph.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Node identifier |
| `node` | `Agent` or `Callable` | Agent or function `(state) -> state` |

### add_edge

```python
def add_edge(from_node: str, to_node: str) -> None
```

Add a direct edge between nodes.

### add_conditional_edge

```python
def add_conditional_edge(from_node: str, condition: Callable[[dict], str]) -> None
```

Add conditional routing. `condition(state)` returns next node name or `END`.

### set_entry

```python
def set_entry(name: str) -> None
```

Set the entry point node.

### run

```python
async def run(prompt: str, state: dict | None = None) -> dict
```

Execute the graph. Returns final state.

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | `str` | Initial prompt (becomes `state["input"]`) |
| `state` | `dict` | Optional initial state |

**Returns:** Final state dict with `input`, `output`, and node outputs.

### run_sync

```python
def run_sync(prompt: str, state: dict | None = None) -> dict
```

Synchronous version of `run()`.

## Constants

### END

```python
from pure_agents import END
```

Use as return value from conditional edges to terminate the graph.

## State

State is a dict that flows through nodes:

```python
{
    "input": "original prompt",
    "output": "current output",
    "node_name": "output from that node",
}
```

## Example

```python
from pure_agents import Agent, Graph, END

researcher = Agent(system="Research the topic.")
writer = Agent(system="Write a summary.")

graph = Graph()
graph.add_node("research", researcher)
graph.add_node("write", writer)
graph.add_edge("research", "write")
graph.set_entry("research")

result = await graph.run("AI trends")
print(result["output"])
```

## Conditional example

```python
def should_continue(state: dict) -> str:
    if "done" in state.get("output", "").lower():
        return END
    return "process"


graph.add_conditional_edge("check", should_continue)
```
