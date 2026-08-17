---
sidebar_position: 13
---

# Validation

Validate agent responses and retry on failure.

## Basic usage

```python
import json


def is_valid_json(response: str) -> bool:
    """Check if response is valid JSON."""
    try:
        json.loads(response)
        return True
    except:
        return False


agent = Agent(
    validator=is_valid_json,
    validation_retries=3,
)

result = await agent.run("Return a JSON object with name and age")
# Guaranteed to be valid JSON (or raises after 3 retries)
```

## How it works

1. Agent generates response
2. Validator function is called with the response
3. If validator returns `True`, response is returned
4. If validator returns `False`:
   - Agent retries with feedback
   - Repeats up to `validation_retries` times
5. If all retries fail, last response is returned

## Validator function

The validator must:
- Accept a single `str` argument (the response)
- Return `bool` (`True` = valid, `False` = invalid)

```python
def validator(response: str) -> bool:
    # Your validation logic
    return True  # or False
```

## Examples

### JSON with required fields

```python
def has_required_fields(response: str) -> bool:
    try:
        data = json.loads(response)
        return "name" in data and "email" in data
    except:
        return False


agent = Agent(validator=has_required_fields, validation_retries=2)
```

### Minimum length

```python
def min_length(response: str) -> bool:
    return len(response) >= 100


agent = Agent(validator=min_length, validation_retries=2)
```

### Contains keywords

```python
def contains_summary(response: str) -> bool:
    return "summary:" in response.lower()


agent = Agent(validator=contains_summary, validation_retries=2)
```

### Format check

```python
import re


def is_email_format(response: str) -> bool:
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return bool(re.match(pattern, response.strip()))


agent = Agent(validator=is_email_format, validation_retries=2)
```

## Combined with structured outputs

For strict type validation, use [structured outputs](/docs/guides/structured-outputs) instead:

```python
from dataclasses import dataclass


@dataclass
class User:
    name: str
    age: int


# This guarantees the correct structure
result = await agent.run("Create a user", output=User)
```

Use validators for:
- Content validation (not just structure)
- Custom business rules
- Format requirements

## Debug mode

See validation attempts with `debug=True`:

```python
agent = Agent(
    validator=is_valid_json,
    validation_retries=3,
    debug=True,
)
# Prints: [Validation failed] Retrying...
```
