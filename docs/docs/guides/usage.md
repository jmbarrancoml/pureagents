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

`cost()` prices the usage against the agent's own model:

```python
agent = Agent(provider="anthropic")
await agent.run("Hello")

print(agent.usage.cost())  # 0.00042
```

Only models with a published rate on file return a number. Anything else
returns `None`, so an estimate is never quietly wrong:

```python
print(Usage(model="some-other-model").cost())  # None
```

Set your own rates in USD per million tokens when the model is not in the
table, or when you are on negotiated pricing:

```python
agent.usage.set_rates(2.00, 6.00)
print(agent.usage.cost())  # priced with your rates
```

You can also price against a specific model without changing the agent:

```python
print(agent.usage.cost("claude-haiku-4-5"))
```

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
    spent = agent.usage.cost()
    if spent is not None and spent > 1.0:
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
