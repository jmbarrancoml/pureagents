"""Hooks - monitor and log agent behaviour."""

import asyncio
from datetime import datetime

from pure_agents import Agent, tool


# Simple logging hooks
def log_tool_call(name: str, args: dict) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] CALL: {name}({args})")


def log_tool_result(name: str, result: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    preview = result[:50] + "..." if len(result) > 50 else result
    print(f"[{timestamp}] RESULT: {name} -> {preview}")


def log_thinking(text: str) -> None:
    preview = text[:80] + "..." if len(text) > 80 else text
    print(f"[THINKING] {preview}")


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for '{query}': Found 5 relevant articles about the topic."


@tool
def summarise(text: str) -> str:
    """Summarise text."""
    return f"Summary: {text[:100]}..."


async def main():
    # Create agent with hooks for monitoring
    agent = Agent(
        tools=[search, summarise],
        on_tool_call=log_tool_call,
        on_tool_result=log_tool_result,
        on_thinking=log_thinking,
    )

    print("=== Agent with Monitoring ===\n")

    response = await agent.run(
        "Search for information about machine learning and summarise it"
    )

    print(f"\n=== Final Response ===\n{response}")


if __name__ == "__main__":
    asyncio.run(main())
