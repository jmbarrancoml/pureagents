"""Memory backends for conversation persistence."""

from __future__ import annotations

import json
import os
import re
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from pure_agents.message import Message

# Session ids become filenames, so anything that could climb out of the
# sessions directory is rejected rather than quietly rewritten.
SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not SAFE_SESSION_ID.match(session_id):
        raise ValueError(
            f"Invalid session id {session_id!r}. Use letters, digits, dots, "
            "dashes and underscores, starting with a letter or digit."
        )
    if ".." in session_id:
        raise ValueError(f"Invalid session id {session_id!r}: '..' is not allowed.")
    return session_id


def write_json_atomically(path: Path, payload: object) -> None:
    """Write via a temp file in the same directory, then rename.

    A crash mid-write used to leave a truncated file that nothing could load.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


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
        return self.path / f"{validate_session_id(session_id)}.json"

    def save(self, session_id: str, messages: list[Message]) -> None:
        data = [m.to_dict() for m in messages]
        write_json_atomically(self._session_path(session_id), data)

    def load(self, session_id: str) -> list[Message] | None:
        path = self._session_path(session_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Session file {path} is not valid JSON: {e}") from e
        return [Message.from_dict(m) for m in data]

    def delete(self, session_id: str) -> None:
        self._session_path(session_id).unlink(missing_ok=True)

    def list_sessions(self) -> list[str]:
        return sorted(p.stem for p in self.path.glob("*.json"))
