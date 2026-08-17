---
sidebar_position: 3
---

# Memory

Memory backends for conversation persistence.

## Memory (base class)

```python
class Memory(ABC):
    @abstractmethod
    def save(self, session_id: str, messages: list[Message]) -> None: ...

    @abstractmethod
    def load(self, session_id: str) -> list[Message] | None: ...

    @abstractmethod
    def delete(self, session_id: str) -> None: ...

    @abstractmethod
    def list_sessions(self) -> list[str]: ...
```

Implement this to create custom storage backends.

### Methods

| Method | Description |
|--------|-------------|
| `save(session_id, messages)` | Save messages for a session |
| `load(session_id)` | Load messages, returns `None` if not found |
| `delete(session_id)` | Delete a session |
| `list_sessions()` | List all session IDs |

## JSONMemory

Default file-based storage.

```python
from pure_agents import JSONMemory

memory = JSONMemory()  # Uses ~/.pureagents/sessions/
memory = JSONMemory("./my-sessions")  # Custom path
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str \| Path \| None` | `~/.pureagents/sessions` | Directory for session files |

### Storage format

Sessions are stored as JSON files:

```
~/.pureagents/sessions/
├── my-session.json
├── chat-1.json
└── chat-2.json
```

Each file contains:

```json
[
  {"role": "system", "content": "You are a helpful assistant..."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

## Example: Custom memory

```python
from pure_agents import Memory, Message
import sqlite3


class SqliteMemory(Memory):
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                messages TEXT
            )
        """)

    def save(self, session_id: str, messages: list[Message]) -> None:
        data = json.dumps([m.to_dict() for m in messages])
        self.conn.execute(
            "INSERT OR REPLACE INTO sessions VALUES (?, ?)", (session_id, data)
        )
        self.conn.commit()

    def load(self, session_id: str) -> list[Message] | None:
        cursor = self.conn.execute(
            "SELECT messages FROM sessions WHERE id = ?", (session_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return [Message.from_dict(m) for m in json.loads(row[0])]

    def delete(self, session_id: str) -> None:
        self.conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self.conn.commit()

    def list_sessions(self) -> list[str]:
        cursor = self.conn.execute("SELECT id FROM sessions")
        return [row[0] for row in cursor.fetchall()]
```

Usage:

```python
agent = Agent(session="my-chat", memory=SqliteMemory("conversations.db"))
```
