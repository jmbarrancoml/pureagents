"""Memory backends for conversation persistence."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from pure_agents.message import Message


class Memory(ABC):
    """Base class for memory backends."""

    @abstractmethod
    def save(self, session_id: str, messages: list[Message]) -> None:
        pass

    @abstractmethod
    def load(self, session_id: str) -> list[Message] | None:
        pass

    @abstractmethod
    def delete(self, session_id: str) -> None:
        pass

    @abstractmethod
    def list_sessions(self) -> list[str]:
        pass


class JSONMemory(Memory):
    """JSON file-based memory."""

    def __init__(self, path: str | Path | None = None):
        if path is None:
            path = Path.home() / ".pureagents" / "sessions"
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        return self.path / f"{session_id}.json"

    def save(self, session_id: str, messages: list[Message]) -> None:
        data = [m.to_dict() for m in messages]
        self._session_path(session_id).write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    def load(self, session_id: str) -> list[Message] | None:
        path = self._session_path(session_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return [Message.from_dict(m) for m in data]

    def delete(self, session_id: str) -> None:
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()

    def list_sessions(self) -> list[str]:
        return [p.stem for p in self.path.glob("*.json")]
