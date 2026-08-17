---
sidebar_position: 3
---

# Memory

Persist conversations across sessions.

## Session-based memory

Just add a session ID:

```python
from pure_agents import Agent

agent = Agent(session="my-chat")
await agent.run("Hi, I'm Jose")

# Later, in a new process...
agent = Agent(session="my-chat")
await agent.run("What's my name?")  # Remembers "Jose"
```

Sessions are saved to `~/.pureagents/sessions/` by default.

## Explicit save/load

For manual control:

```python
agent = Agent()
await agent.run("Hello")
await agent.run("How are you?")

# Save to file
agent.save("conversation.json")

# Load in a new agent
agent2 = Agent()
agent2.load("conversation.json")
await agent2.run("What did I say first?")  # Remembers
```

## Clear history

```python
agent.clear()  # Clears messages and deletes session file
```

## Custom storage backend

Implement the `Memory` interface:

```python
from pure_agents import Memory, Message


class RedisMemory(Memory):
    def __init__(self, redis_url: str):
        self.client = redis.from_url(redis_url)

    def save(self, session_id: str, messages: list[Message]) -> None:
        data = [m.to_dict() for m in messages]
        self.client.set(f"session:{session_id}", json.dumps(data))

    def load(self, session_id: str) -> list[Message] | None:
        data = self.client.get(f"session:{session_id}")
        if not data:
            return None
        return [Message.from_dict(m) for m in json.loads(data)]

    def delete(self, session_id: str) -> None:
        self.client.delete(f"session:{session_id}")

    def list_sessions(self) -> list[str]:
        keys = self.client.keys("session:*")
        return [k.decode().replace("session:", "") for k in keys]


# Use it
agent = Agent(session="my-chat", memory=RedisMemory("redis://localhost:6379"))
```

## JSONMemory options

The default backend can use a custom path:

```python
from pure_agents import Agent, JSONMemory

memory = JSONMemory("./my-sessions")
agent = Agent(session="chat-1", memory=memory)
```

## List sessions

```python
from pure_agents import JSONMemory

memory = JSONMemory()
sessions = memory.list_sessions()
print(sessions)  # ["my-chat", "chat-1", ...]
```
