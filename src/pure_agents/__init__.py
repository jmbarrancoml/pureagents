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
    Router,
    Usage,
    chain,
    chain_sync,
    clear_cache,
    completion,
    completion_sync,
    set_cache_size,
)
from pure_agents.memory import JSONMemory, Memory
from pure_agents.message import Message
from pure_agents.tool import Tool, tool

__version__ = "0.6.0"
__all__ = [
    "Agent",
    "END",
    "Graph",
    "JSONMemory",
    "Memory",
    "Message",
    "Router",
    "TEMPLATES",
    "Tool",
    "Usage",
    "chain",
    "chain_sync",
    "clear_cache",
    "completion",
    "completion_sync",
    "set_cache_size",
    "tool",
]
