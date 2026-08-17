---
sidebar_position: 10
---

# Batch processing

Run multiple prompts in parallel.

## Basic usage

```python
agent = Agent()

results = await agent.batch(
    [
        "What is the capital of France?",
        "What is 2 + 2?",
        "Who wrote Hamlet?",
    ]
)

for result in results:
    print(result)
```

## How it works

Each prompt runs in a **fresh agent instance** with the same config. Results return in the same order as inputs.

## Performance

Batch is much faster than sequential execution:

```python
import time

prompts = ["Q1", "Q2", "Q3", "Q4", "Q5"]

# Sequential: ~10 seconds
start = time.time()
for p in prompts:
    await agent.run(p)
print(f"Sequential: {time.time() - start:.1f}s")

# Batch: ~2 seconds
start = time.time()
await agent.batch(prompts)
print(f"Batch: {time.time() - start:.1f}s")
```

## Usage tracking

Token usage is aggregated across all batch requests:

```python
await agent.batch(["Q1", "Q2", "Q3"])
print(agent.usage.total_tokens)  # Combined total
```

## With tools

Tools work in batch mode:

```python
@tool
def search(query: str) -> str:
    return f"Results for {query}"


agent = Agent(tools=[search])
results = await agent.batch(
    [
        "Search for Python tutorials",
        "Search for Rust tutorials",
    ]
)
```

## When to use

- Processing multiple independent queries
- Bulk data extraction
- Parallel API calls
- Any time you have multiple prompts that don't depend on each other
