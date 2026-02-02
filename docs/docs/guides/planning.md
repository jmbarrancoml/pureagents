---
sidebar_position: 16
---

# Planning

Make the agent create a plan before executing.

## Basic usage

```python
from pure_agents import Agent, tool

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for {query}..."

agent = Agent(tools=[search])

# With planning
result = await agent.run(
    "Research and summarise the history of Python",
    plan=True,
)
```

## How it works

With `plan=True`, the agent:

1. **Plans**: Creates a numbered step-by-step plan
2. **Executes**: Runs through the plan using tools as needed

```
prompt → create plan → execute plan with tools → result
```

## Monitor the plan

Use `on_plan` hook to see the plan:

```python
def show_plan(plan: str):
    print(f"Plan:\n{plan}")

agent = Agent(
    tools=[search, write],
    on_plan=show_plan,
)

await agent.run("Build a report on AI trends", plan=True)
```

## When to use planning

**Good for:**
- Complex, multi-step tasks
- Tasks requiring research then synthesis
- When you want visibility into the agent's approach

**Not needed for:**
- Simple questions
- Single tool calls
- Quick lookups

## Example output

```
[Plan]
1. Search for information about Python's creation
2. Search for major Python versions and their features
3. Search for Python's current usage statistics
4. Write a summary combining all findings

[Step 1/10]
[Call] search({"query": "Python programming language history creation"})
[Result] Python was created by Guido van Rossum...
...
[Final] Python is a programming language created in 1991...
```

## Combine with other features

```python
result = await agent.run(
    "Analyse market trends",
    plan=True,
    output=MarketReport,  # Structured output
)
```
