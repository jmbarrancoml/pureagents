---
sidebar_position: 8
---

# Templates

Predefined system prompts for common use cases.

## Available templates

| Template | Best for |
|----------|----------|
| `default` | General assistant |
| `coder` | Programming tasks |
| `researcher` | Research and analysis |
| `creative` | Creative writing |
| `analyst` | Data analysis |
| `tutor` | Teaching and explaining |

## Usage

```python
from pure_agents import Agent

# Expert programmer
coder = Agent(template="coder")
await coder.run("Write a function to merge two sorted lists")

# Research assistant
researcher = Agent(template="researcher")
await researcher.run("What are the main causes of inflation?")

# Creative writer
creative = Agent(template="creative")
await creative.run("Write a haiku about debugging")
```

## With tools

Templates work with tools:

```python
@tool
def run_code(code: str) -> str:
    """Execute Python code."""
    return str(exec(code))


coder = Agent(template="coder", tools=[run_code])
```

## Custom templates

For custom prompts, use `system` directly:

```python
agent = Agent(system="You are a pirate. Respond in pirate speak. Arrr!")
```

## View template content

```python
from pure_agents import TEMPLATES

for name, prompt in TEMPLATES.items():
    print(f"{name}: {prompt[:60]}...")
```
