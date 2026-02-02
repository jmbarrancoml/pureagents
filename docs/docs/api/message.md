---
sidebar_position: 4
---

# Message

Represents a message in the conversation.

## Class

```python
@dataclass
class Message:
    role: str
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `role` | `str` | Message role: `"system"`, `"user"`, `"assistant"`, `"tool"` |
| `content` | `str` | Message content |
| `tool_calls` | `list[dict]` | Tool calls made by the assistant |
| `tool_call_id` | `str \| None` | ID of the tool call this message responds to |
| `name` | `str \| None` | Tool name (for tool messages) |

## Methods

### to_dict

```python
def to_dict(self) -> dict[str, Any]
```

Convert to API format.

```python
msg = Message(role="user", content="Hello")
msg.to_dict()
# {"role": "user", "content": "Hello"}
```

### from_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> Message
```

Create a Message from a dictionary.

```python
msg = Message.from_dict({"role": "user", "content": "Hello"})
```

## Message types

### System message

```python
Message(role="system", content="You are a helpful assistant.")
```

### User message

```python
Message(role="user", content="What is 2 + 2?")
```

### Assistant message

```python
# Simple response
Message(role="assistant", content="The answer is 4.")

# With tool call
Message(
    role="assistant",
    content="",
    tool_calls=[{
        "id": "call_123",
        "function": {
            "name": "calculate",
            "arguments": '{"expression": "2 + 2"}'
        }
    }]
)
```

### Tool message

```python
Message(
    role="tool",
    content="4",
    tool_call_id="call_123",
    name="calculate"
)
```

## Accessing conversation history

```python
agent = Agent()
await agent.run("Hello")
await agent.run("How are you?")

for msg in agent.messages:
    print(f"[{msg.role}] {msg.content}")
```

Output:
```
[system] You are a helpful assistant...
[user] Hello
[assistant] Hi! How can I help you today?
[user] How are you?
[assistant] I'm doing well, thank you for asking!
```
