---
sidebar_position: 6
---

# Structured outputs

Get typed responses instead of raw strings.

## Basic usage

Define a dataclass and pass it to `run()`:

```python
from dataclasses import dataclass
from pure_agents import Agent


@dataclass
class Sentiment:
    sentiment: str
    confidence: float
    summary: str


agent = Agent()
result = await agent.run("Analyse: I absolutely love this product!", output=Sentiment)

print(result.sentiment)  # "positive"
print(result.confidence)  # 0.95
print(result.summary)  # "Strong positive sentiment..."
```

## Supported types

Dataclass fields can be:

- `str`: strings
- `int`: integers
- `float`: numbers
- `bool`: booleans

```python
@dataclass
class Recipe:
    name: str
    prep_time_minutes: int
    is_vegetarian: bool
    rating: float
```

## How it works

pureagents automatically:

1. Converts your dataclass to a JSON schema
2. Adds instructions to the prompt asking for JSON output
3. Parses the response into your dataclass

No magic. You can see exactly what's happening with `debug=True`.

## Error handling

If the LLM returns malformed JSON:

```python
try:
    result = await agent.run("...", output=MyClass)
except json.JSONDecodeError:
    # Handle invalid JSON
    pass
```

## When to use

- Extracting structured data from text
- Building APIs that need typed responses
- Data pipelines with validated outputs
- Any time you want guarantees about response format
