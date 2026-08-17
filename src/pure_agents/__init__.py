"""
Pure Agents - LLM agents without the complexity.

Example:
    from pure_agents import Agent, tool

    @tool
    def search(query: str) -> str:
        '''Search the web.'''
        return f"Results for {query}"

    agent = Agent(tools=[search])
    result = await agent.run("Find the weather in Madrid")
"""

from pure_agents.agent import (
    END,
    TEMPLATES,
    Agent,
    Graph,
    MaxStepsError,
    Router,
    StreamEvent,
    StructuredOutputError,
    Usage,
    chain,
    chain_sync,
    clear_cache,
    completion,
    completion_sync,
    set_cache_size,
)
from pure_agents.clients import TruncatedResponseError
from pure_agents.memory import JSONMemory, Memory
from pure_agents.message import Message
from pure_agents.tool import Tool, tool

__version__ = "0.6.0"
__all__ = [
    "Agent",
    "END",
    "Graph",
    "JSONMemory",
    "MaxStepsError",
    "Memory",
    "Message",
    "Router",
    "StreamEvent",
    "StructuredOutputError",
    "TEMPLATES",
    "Tool",
    "TruncatedResponseError",
    "Usage",
    "chain",
    "chain_sync",
    "clear_cache",
    "completion",
    "completion_sync",
    "set_cache_size",
    "tool",
]
