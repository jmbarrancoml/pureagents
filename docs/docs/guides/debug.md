---
sidebar_position: 5
---

# Debug mode

See what your agent is doing internally.

## Enable debug

```python
from pure_agents import Agent, tool


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    return str(eval(expression))


agent = Agent(tools=[calculate], debug=True)
result = await agent.run("What is 25 * 4?")
```

Output:
```
[Step 1/10]
[Call] calculate({'expression': '25 * 4'})
[Result] 100
[Step 2/10]
[Final] The answer is 100.
```

## What debug shows

| Output | Meaning |
|--------|---------|
| `[Step N/M]` | Current step in the ReAct loop |
| `[Call] tool(args)` | Tool being called with arguments |
| `[Result] ...` | Tool return value |
| `[Final] ...` | Final response (non-streaming) |

## Debug with streaming

Works with streaming too:

```python
agent = Agent(tools=[calculate], debug=True)

async for event in agent.stream("Calculate 100 / 4"):
    if event.type == "text":
        print(event.content, end="", flush=True)
```

## Max steps

The agent stops after `max_steps` to prevent infinite loops:

```python
agent = Agent(max_steps=5, debug=True)
```

Running out of steps raises `MaxStepsError`, so it can't be mistaken for an
answer. The last thing the model said is on `.partial`:

```python
from pure_agents import MaxStepsError

try:
    result = await agent.run("Keep calling tools forever")
except MaxStepsError as e:
    print(f"Gave up after {e.steps} steps. Last said: {e.partial}")
```

`stream()` ends with a `done` event instead, since events have already reached
the caller by then.

## Inspecting messages

Access the conversation history directly:

```python
agent = Agent()
await agent.run("Hello")

for msg in agent.messages:
    print(f"{msg.role}: {msg.content}")
    if msg.tool_calls:
        print(f"  Tool calls: {msg.tool_calls}")
```
