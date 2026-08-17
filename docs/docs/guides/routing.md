---
sidebar_position: 15
---

# Routing

Direct prompts to different agents based on content.

## Function-based routing

You define the routing logic:

```python
from pure_agents import Agent, Router

coder = Agent(template="coder")
writer = Agent(template="creative")
default = Agent()


def route(prompt: str) -> str:
    if "code" in prompt.lower():
        return "coder"
    if "write" in prompt.lower() or "story" in prompt.lower():
        return "writer"
    return "default"


router = Router(
    agents={"coder": coder, "writer": writer, "default": default},
    route=route,
)

result = await router.run("Write a Python function for sorting")
```

## LLM-based routing

Let the LLM decide which agent to use:

```python
router = Router(
    agents={
        "coder": coder,
        "writer": writer,
        "analyst": analyst,
    },
    # No route function = LLM decides
)

result = await router.run("Explain the pros and cons of microservices")
```

The router uses a small LLM call to classify the prompt and select the appropriate agent.

## Custom LLM router

Configure the routing LLM:

```python
router = Router(
    agents={"a": agent_a, "b": agent_b},
    provider="openai",
    model="gpt-4o-mini",
)
```

## Get the selected agent

```python
agent_name = await router.route("Your prompt")
print(f"Selected: {agent_name}")
```

## Sync version

```python
result = router.run_sync("Your prompt")
```

## Use cases

**Support system:**
```python
router = Router(
    agents={
        "billing": billing_agent,
        "technical": tech_agent,
        "general": general_agent,
    },
    route=classify_support_ticket,
)
```

**Multi-language:**
```python
router = Router(
    agents={
        "spanish": spanish_agent,
        "english": english_agent,
        "french": french_agent,
    },
    route=detect_language,
)
```

**Complexity-based:**
```python
def route_by_complexity(prompt: str) -> str:
    if len(prompt) > 500 or "explain" in prompt.lower():
        return "detailed"
    return "quick"


router = Router(
    agents={"detailed": thorough_agent, "quick": fast_agent},
    route=route_by_complexity,
)
```
