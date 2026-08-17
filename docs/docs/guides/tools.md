---
sidebar_position: 1
---

# Tools

Tools let your agent interact with the world.

## The @tool decorator

The simplest way to create a tool:

```python
from pure_agents import tool


@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"
```

The decorator extracts:
- **Name** from the function name
- **Description** from the docstring
- **Parameters** from type hints

## Type hints

pureagents converts Python types to JSON Schema:

```python
@tool
def calculate(a: int, b: float, operation: str) -> float:
    """Perform a calculation."""
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
```

| Python type | JSON Schema |
|-------------|-------------|
| `str` | `string` |
| `int` | `integer` |
| `float` | `number` |
| `bool` | `boolean` |
| `list` | `array` |
| `dict` | `object` |

## Async tools

Tools can be async:

```python
@tool
async def fetch_data(url: str) -> str:
    """Fetch data from a URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text
```

## Multiple tools

Pass multiple tools to the agent:

```python
agent = Agent(tools=[search, calculate, fetch_data])
```

The agent decides which tool to use based on the user's request.

## Tool class

For more control, create a Tool directly:

```python
from pure_agents import Tool


def my_function(x: str) -> str:
    return x.upper()


my_tool = Tool(
    name="uppercase",
    description="Convert text to uppercase",
    parameters={"x": {"type": "string", "description": "Text to convert"}},
    fn=my_function,
)

agent = Agent(tools=[my_tool])
```

## Tool timeout

Set per-tool execution limits:

```python
@tool(timeout=10)
async def slow_search(query: str) -> str:
    """Search with 10-second limit."""
    # If this takes > 10s, raises TimeoutError
    return await external_api.search(query)
```

## Tool groups

Organise tools and enable/disable them dynamically:

```python
@tool(group="web")
def search(query: str) -> str:
    """Search the web."""
    return f"Results for {query}"


@tool(group="web")
def fetch(url: str) -> str:
    """Fetch a URL."""
    return f"Content of {url}"


@tool(group="math")
def calculate(expr: str) -> str:
    """Evaluate expression."""
    return str(eval(expr))


# Start with only web tools enabled
agent = Agent(
    tools=[search, fetch, calculate],
    enabled_groups=["web"],
)

# Toggle groups at runtime
agent.enable_group("math")
agent.disable_group("web")
```

## Tool choice

Control how the agent uses tools:

```python
# Let agent decide (default)
agent = Agent(tools=[...], tool_choice="auto")

# Force tool use
agent = Agent(tools=[...], tool_choice="required")

# Disable tools for this agent
agent = Agent(tools=[...], tool_choice="none")
```

## Parallel execution

When the LLM returns multiple tool calls, they run in parallel:

```python
@tool
async def fetch_weather(city: str) -> str:
    await asyncio.sleep(1)  # Simulated API
    return f"Weather in {city}: sunny"


@tool
async def fetch_news(topic: str) -> str:
    await asyncio.sleep(1)  # Simulated API
    return f"News about {topic}: 5 articles"


agent = Agent(tools=[fetch_weather, fetch_news])

# Both tools run in parallel (~1s, not ~2s)
result = await agent.run("Get weather in Madrid and news about AI")
```

## Best practices

1. **Clear docstrings**: The LLM uses them to decide when to call the tool
2. **Type hints**: Always include them for proper schema generation
3. **Simple return values**: Return strings or values that convert cleanly to strings
4. **Handle errors**: Wrap tool logic in try/except if needed
5. **Use timeouts**: Protect against slow external services
6. **Use groups**: Organise related tools for easy management
