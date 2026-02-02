---
sidebar_position: 2
---

# Tool

Tools allow agents to perform actions.

## @tool decorator

```python
def tool(
    fn: Callable[..., Any] | None = None,
    *,
    timeout: float | None = None,
    group: str | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]
```

Create a tool from a function.

### Basic usage

```python
from pure_agents import tool

@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"
```

### With parameters

```python
@tool(timeout=10, group="web")
def search(query: str) -> str:
    """Search with timeout and grouping."""
    return f"Results for {query}"
```

### Decorator parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout` | `float` | `None` | Max execution time (seconds) |
| `group` | `str` | `None` | Group name for enable/disable |

The decorator extracts:
- Name from `fn.__name__`
- Description from `fn.__doc__`
- Parameters from type hints

## Tool class

```python
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    timeout: float | None = None
    group: str | None = None
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | Required | Tool name (used by the LLM) |
| `description` | `str` | Required | What the tool does |
| `parameters` | `dict[str, Any]` | Required | JSON Schema for parameters |
| `fn` | `Callable` | Required | Function to execute |
| `timeout` | `float` | `None` | Max execution time (seconds) |
| `group` | `str` | `None` | Group name for enable/disable |

### Methods

#### to_dict

```python
def to_dict(self) -> dict[str, Any]
```

Convert to OpenAI/Mistral tool format.

```python
tool_dict = my_tool.to_dict()
# {
#     "type": "function",
#     "function": {
#         "name": "greet",
#         "description": "Greet someone by name.",
#         "parameters": {
#             "type": "object",
#             "properties": {"name": {"type": "string"}},
#             "required": ["name"]
#         }
#     }
# }
```

#### call

```python
async def call(self, **kwargs: Any) -> str
```

Execute the tool function.

```python
result = await my_tool.call(name="World")
# "Hello, World!"
```

## Manual tool creation

```python
from pure_agents import Tool

def uppercase(text: str) -> str:
    return text.upper()

my_tool = Tool(
    name="uppercase",
    description="Convert text to uppercase",
    parameters={
        "text": {
            "type": "string",
            "description": "Text to convert"
        }
    },
    fn=uppercase,
    timeout=5.0,
    group="text",
)
```

## Tool timeout

Tools can have individual timeouts:

```python
@tool(timeout=10)
async def slow_api(query: str) -> str:
    """Call slow external API."""
    await asyncio.sleep(5)
    return f"Result: {query}"
```

If the tool exceeds its timeout, it raises `asyncio.TimeoutError`.

## Tool groups

Groups let you enable/disable sets of tools dynamically:

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
    """Calculate expression."""
    return str(eval(expr))

# Only enable web tools
agent = Agent(
    tools=[search, fetch, calculate],
    enabled_groups=["web"],
)

# Or toggle at runtime
agent.enable_group("math")
agent.disable_group("web")
```

## Type mapping

| Python | JSON Schema |
|--------|-------------|
| `str` | `{"type": "string"}` |
| `int` | `{"type": "integer"}` |
| `float` | `{"type": "number"}` |
| `bool` | `{"type": "boolean"}` |
| `list` | `{"type": "array"}` |
| `dict` | `{"type": "object"}` |
