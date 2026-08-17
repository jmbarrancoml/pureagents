---
sidebar_position: 17
---

# Graph

Multi-agent workflows with conditional routing. Similar to LangGraph but simpler.

## Basic usage

```python
from pure_agents import Agent, Graph, END

researcher = Agent(system="Research the topic.")
writer = Agent(system="Write a summary.")

graph = Graph()
graph.add_node("research", researcher)
graph.add_node("write", writer)
graph.add_edge("research", "write")
graph.set_entry("research")

result = await graph.run("AI trends in 2024")
print(result["output"])
```

## How it works

```
entry → node1 → node2 → ... → END
              ↘ conditional → node3
```

- State flows through nodes as a dict
- Each node modifies state
- Edges define the flow
- Conditional edges decide next node based on state

## State

State is a dict that flows through the graph:

```python
{
    "input": "original prompt",
    "output": "current output",
    "node_name": "output from that node",
    # ... your custom keys
}
```

## Conditional edges

Route based on state:

```python
def should_revise(state: dict) -> str:
    if "error" in state.get("output", ""):
        return "fix"  # Go to fix node
    return END  # Finish


graph.add_conditional_edge("review", should_revise)
```

## Function nodes

Nodes can be functions instead of agents:

```python
def process(state: dict) -> dict:
    state["output"] = state["input"].upper()
    return state


graph.add_node("process", process)
```

## Example: research pipeline with review

```python
from pure_agents import Agent, Graph, END

researcher = Agent(system="Find facts about the topic.")
writer = Agent(system="Write a clear summary.")
reviewer = Agent(system="Review. Say APPROVED or NEEDS_REVISION.")

graph = Graph()

graph.add_node("research", researcher)
graph.add_node("write", writer)
graph.add_node("review", reviewer)

graph.add_edge("research", "write")
graph.add_edge("write", "review")


def check_review(state: dict) -> str:
    if "NEEDS_REVISION" in state.get("review", ""):
        return "write"  # Loop back
    return END


graph.add_conditional_edge("review", check_review)
graph.set_entry("research")

result = await graph.run("Climate change solutions")
```

## Accessing intermediate results

```python
result = await graph.run("My prompt")

print(result["input"])  # Original prompt
print(result["output"])  # Final output
print(result["research"])  # Output from research node
print(result["write"])  # Output from write node
```

## Sync version

```python
result = graph.run_sync("My prompt")
```

## Tips

- Keep graphs simple. If you need complex logic, use functions as nodes.
- Use `END` to terminate the graph.
- Maximum 100 steps to prevent infinite loops.
- State is mutable. Each node can read and modify it.
