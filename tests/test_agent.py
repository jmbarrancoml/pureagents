"""Tests for pure_agents."""

import tempfile
from pathlib import Path

import pytest

from pure_agents import Agent, JSONMemory, Message, Tool, tool


class TestToolDecorator:
    """Test the @tool decorator."""

    def test_creates_tool_from_function(self):
        @tool
        def greet(name: str) -> str:
            """Say hello to someone."""
            return f"Hello, {name}!"

        assert isinstance(greet, Tool)
        assert greet.name == "greet"
        assert greet.description == "Say hello to someone."
        assert "name" in greet.parameters

    def test_extracts_type_hints(self):
        @tool
        def calculate(a: int, b: float) -> float:
            """Add two numbers."""
            return a + b

        assert calculate.parameters["a"]["type"] == "integer"
        assert calculate.parameters["b"]["type"] == "number"

    def test_to_dict_format(self):
        @tool
        def search(query: str) -> str:
            """Search the web."""
            return query

        d = search.to_dict()
        assert d["type"] == "function"
        assert d["function"]["name"] == "search"
        assert d["function"]["description"] == "Search the web."


class TestToolCall:
    """Test tool execution."""

    @pytest.mark.asyncio
    async def test_sync_function(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        result = await add.call(a=2, b=3)
        assert result == "5"

    @pytest.mark.asyncio
    async def test_async_function(self):
        @tool
        async def async_greet(name: str) -> str:
            """Greet async."""
            return f"Hello, {name}!"

        result = await async_greet.call(name="World")
        assert result == "Hello, World!"


class TestAgent:
    """Test Agent class."""

    def test_requires_api_key(self):
        with pytest.raises(ValueError, match="API key required"):
            Agent()

    def test_registers_tools(self):
        @tool
        def my_tool(x: str) -> str:
            """A tool."""
            return x

        agent = Agent(tools=[my_tool], api_key="test-key")
        assert "my_tool" in agent.tools

    def test_custom_system(self):
        agent = Agent(
            api_key="test-key",
            system="You are a pirate.",
        )
        assert agent.system == "You are a pirate."

    def test_openai_provider(self):
        agent = Agent(api_key="test-key", provider="openai")
        assert agent.provider == "openai"
        assert agent.model == "gpt-5.2-instant"
        assert agent.client.base_url == "https://api.openai.com/v1"

    def test_invalid_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            Agent(api_key="test-key", provider="invalid")

    def test_anthropic_provider(self):
        agent = Agent(api_key="test-key", provider="anthropic")
        assert agent.provider == "anthropic"
        assert agent.model == "claude-opus-5"
        assert agent.client.base_url == "https://api.anthropic.com/v1"


class TestMemory:
    """Test memory functionality."""

    def test_json_memory_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = JSONMemory(tmpdir)
            messages = [
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
            ]

            memory.save("test-session", messages)
            loaded = memory.load("test-session")

            assert loaded is not None
            assert len(loaded) == 2
            assert loaded[0].content == "Hello"
            assert loaded[1].content == "Hi there!"

    def test_json_memory_load_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = JSONMemory(tmpdir)
            assert memory.load("nonexistent") is None

    def test_json_memory_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = JSONMemory(tmpdir)
            messages = [Message(role="user", content="test")]

            memory.save("to-delete", messages)
            assert memory.load("to-delete") is not None

            memory.delete("to-delete")
            assert memory.load("to-delete") is None

    def test_json_memory_list_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = JSONMemory(tmpdir)
            memory.save("session-1", [Message(role="user", content="1")])
            memory.save("session-2", [Message(role="user", content="2")])

            sessions = memory.list_sessions()
            assert "session-1" in sessions
            assert "session-2" in sessions

    def test_agent_save_load_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = Agent(api_key="test-key")
            agent.messages = [
                Message(role="system", content="You are helpful."),
                Message(role="user", content="Hello"),
            ]

            path = Path(tmpdir) / "chat.json"
            agent.save(str(path))

            agent2 = Agent(api_key="test-key")
            agent2.load(str(path))

            assert len(agent2.messages) == 2
            assert agent2.messages[1].content == "Hello"

    def test_agent_with_session_creates_memory(self):
        agent = Agent(api_key="test-key", session="my-session")
        assert agent.memory is not None
        assert agent.session == "my-session"
