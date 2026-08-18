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

| Python | JSON Schema |
|--------|-------------|
| `str`, `int`, `float`, `bool` | the matching primitive |
| `list[T]` | array of `T` |
| `dict[str, T]` | object with `T` values |
| `T \| None`, `Optional[T]` | `T`, nullable |
| `Literal["a", "b"]` | enum |
| `Enum` subclass | enum of its values |
| another dataclass | nested object |

```python
from typing import Literal


@dataclass
class Recipe:
    name: str
    prep_time_minutes: int
    is_vegetarian: bool
    ingredients: list[str]
    difficulty: Literal["easy", "medium", "hard"]
    notes: str | None = None
```

## How it works

The provider enforces the schema while it decodes, so the reply cannot come
back the wrong shape:

1. Your dataclass becomes a JSON schema.
2. The schema is tightened for strict mode: every object forbids extra
   properties, and every property is listed as required. A field with a default
   becomes nullable, because that is how strict mode expresses "may be missing".
3. The schema travels in the request's own field, not in the prompt.
   OpenAI-compatible endpoints receive `response_format.json_schema` with
   `strict: true`; Anthropic receives `output_config.format`.
4. The reply is parsed into your dataclass.

Your prompt stays about the task. Run with `debug=True` to see the request.

### Endpoints that cannot enforce a schema

A provider registered with `structured_outputs=False` falls back to describing
the schema in the prompt and parsing whatever comes back:

```python
register_provider(
    "legacy-gateway",
    base_url="https://gateway.internal/v1",
    structured_outputs=False,
)
```

Use it when a server rejects `response_format`. Built-in providers and ad-hoc
`base_url` endpoints are assumed capable, which is true of Ollama, LM Studio and
vLLM on current releases.

## With tools

Tools and `output=` compose. The agent runs its tool loop as usual and the final
answer arrives in your shape:

```python
result = await agent.run("Look up the reviews and summarise", output=Sentiment)
```

The schema is attached to every request in the run, so the provider has to
support constraining the response format alongside tool calls. All three
built-in providers do. If a custom endpoint does not, it will say so with a 400.

## Error handling

A reply that will not fit your dataclass raises `StructuredOutputError`, which
carries the raw text:

```python
from pure_agents import StructuredOutputError

try:
    result = await agent.run("...", output=Sentiment)
except StructuredOutputError as e:
    print(f"Model said: {e.raw}")
```

With provider-side enforcement this should not happen for schema reasons. It
still can when the response was cut short at `max_tokens`, or on an endpoint
using the prompt fallback.

## When to use

- Extracting structured data from text
- Building APIs that need typed responses
- Data pipelines with validated outputs
- Any time you want guarantees about response format
