---
sidebar_position: 11
---

# Usage tracking

Monitor token consumption and estimate costs.

## Basic usage

Every agent tracks usage automatically:

```python
agent = Agent()

await agent.run("Hello, how are you?")
await agent.run("Tell me more")

print(agent.usage.input_tokens)  # Prompt tokens
print(agent.usage.output_tokens)  # Response tokens
print(agent.usage.total_tokens)  # Combined
print(agent.usage.requests)  # Number of API calls
```

## Cost estimation

Get estimated cost in USD:

```python
print(agent.usage.cost("mistral"))  # $0.0001
print(agent.usage.cost("openai"))  # $0.0025
print(agent.usage.cost("anthropic"))  # $0.0030
```

Costs are approximations based on public pricing.

## Reset counters

```python
agent.usage.reset()
print(agent.usage.total_tokens)  # 0
```

## With batch

Batch aggregates usage from all parallel requests:

```python
await agent.batch(["Q1", "Q2", "Q3", "Q4", "Q5"])
print(agent.usage.requests)  # 5
print(agent.usage.total_tokens)  # Combined from all 5
```

## Monitoring example

```python
agent = Agent()

# Run some tasks
for task in tasks:
    await agent.run(task)

    # Check budget
    if agent.usage.cost("openai") > 1.0:
        print("Budget exceeded!")
        break
```

## Usage object

```python
from pure_agents import Usage

usage = Usage()
usage.add(input_tokens=100, output_tokens=50)
print(usage.total_tokens)  # 150
```
