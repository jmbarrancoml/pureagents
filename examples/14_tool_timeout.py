"""Tool timeout - prevent slow tools from hanging."""

import asyncio
import time

from pure_agents import Agent, tool


@tool(timeout=2.0)  # Max 2 seconds
def fast_search(query: str) -> str:
    """Fast search that completes quickly."""
    return f"Results for {query}"


@tool(timeout=2.0)  # Max 2 seconds
def slow_search(query: str) -> str:
    """Slow search that will timeout."""
    time.sleep(5)  # This will timeout!
    return f"Results for {query}"


async def main():
    agent = Agent(tools=[fast_search, slow_search], debug=True)

    # Fast search works fine
    print("=== Fast Search ===")
    result = await agent.run("Use fast_search for 'python tutorials'")
    print(f"Result: {result}")

    # Slow search times out gracefully
    print("\n=== Slow Search (will timeout) ===")
    agent2 = Agent(tools=[slow_search], debug=True)
    result = await agent2.run("Use slow_search for 'rust tutorials'")
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
