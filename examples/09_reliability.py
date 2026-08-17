"""Reliability features - retry, timeout, and context limits."""

import asyncio

from pure_agents import Agent, tool


@tool
def slow_search(query: str) -> str:
    """A slow search that might timeout."""
    import time

    time.sleep(2)  # Simulate slow API
    return f"Results for '{query}'"


async def main():
    # === Retry with exponential backoff ===
    print("=== Retry Example ===")
    agent_with_retries = Agent(
        retries=3,  # Retry up to 3 times on failure
        debug=True,
    )
    # If the API fails, it will retry with 1s, 2s, 4s delays
    response = await agent_with_retries.run("What is the capital of Spain?")
    print(f"Response: {response}")

    # === Timeout ===
    print("\n=== Timeout Example ===")
    agent_with_timeout = Agent(
        timeout=30.0,  # Max 30 seconds per request
    )
    response = await agent_with_timeout.run("Quick question: what is 2+2?")
    print(f"Response: {response}")

    # === Context limit ===
    print("\n=== Context Limit Example ===")
    agent_limited = Agent(
        max_messages=10,  # Keep only last 10 messages
        session="context-demo",
    )

    # Simulate a long conversation
    for i in range(15):
        await agent_limited.run(f"Message number {i + 1}")

    # Only the last 10 messages are kept (plus system prompt)
    print(f"Messages in memory: {len(agent_limited.messages)}")

    # Clean up
    agent_limited.clear()

    # === All together ===
    print("\n=== Combined Example ===")
    robust_agent = Agent(
        retries=2,
        timeout=60.0,
        max_messages=20,
        debug=True,
    )
    response = await robust_agent.run("Hello, how are you?")
    print(f"Response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
