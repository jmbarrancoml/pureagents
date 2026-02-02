---
sidebar_position: 9
---

# Reliability

Handle failures gracefully with retry, timeout, fallback, caching, and context limits.

## Retry with backoff

Automatically retry failed requests with exponential backoff:

```python
agent = Agent(retries=3)
# On failure: wait 1s, retry, wait 2s, retry, wait 4s, retry
```

Useful for:
- Rate limit errors (429)
- Temporary API outages
- Network issues

## Timeout

Set maximum time for each request:

```python
agent = Agent(timeout=30.0)  # 30 seconds max
```

Raises `asyncio.TimeoutError` if exceeded.

## Context limit

Auto-trim old messages to manage context window:

```python
agent = Agent(max_messages=20)
# Keeps system prompt + last 19 messages
```

Essential for:
- Long conversations
- Avoiding context overflow errors
- Controlling costs

## Fallback provider

Automatic failover to another provider:

```python
agent = Agent(
    provider="openai",
    fallback="mistral",  # If OpenAI fails, try Mistral
)
```

The fallback activates when:
- Primary provider returns an error
- All retries are exhausted (if `retries` is set)

You can combine with retries:

```python
agent = Agent(
    provider="openai",
    fallback="mistral",
    retries=2,  # Retry OpenAI twice, then try Mistral
)
```

## Caching

Cache responses to avoid repeated API calls:

```python
from pure_agents import Agent, clear_cache

agent = Agent(cache=True)

# First call hits API
result1 = await agent.run("What is 2+2?")

# Same prompt returns cached response instantly
result2 = await agent.run("What is 2+2?")

# Clear the cache if needed
clear_cache()
```

Cache keys are based on:
- Provider and model
- Messages (prompts)
- Tools configuration

Useful for:
- Development and testing
- Repeated queries
- Cost reduction

## Tool timeout

Per-tool execution limits:

```python
@tool(timeout=10)
async def slow_api(query: str) -> str:
    """Call slow external API with 10s limit."""
    return await external_service.call(query)

agent = Agent(tools=[slow_api])
```

Raises `asyncio.TimeoutError` if exceeded.

## Combined usage

```python
agent = Agent(
    provider="openai",
    fallback="mistral",
    retries=2,
    timeout=60.0,
    max_messages=50,
    cache=True,
)
```

## Debug mode

See retry attempts with `debug=True`:

```python
agent = Agent(retries=3, debug=True)
# Prints: [Retry 1] Connection error. Waiting 1s...
```

## Error handling

```python
from asyncio import TimeoutError

try:
    result = await agent.run("Complex task")
except TimeoutError:
    print("Request timed out")
except Exception as e:
    print(f"All retries failed: {e}")
```
