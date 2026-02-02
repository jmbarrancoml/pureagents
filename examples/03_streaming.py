"""Streaming responses - real-time output."""

import asyncio

from pure_agents import Agent, tool


@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Found 3 results for '{query}': [Article 1], [Article 2], [Article 3]"


async def main():
    agent = Agent(tools=[search_web])

    print("Agent: ", end="", flush=True)

    # Stream the response token by token
    async for chunk in agent.stream("Tell me about the history of Python"):
        print(chunk, end="", flush=True)

    print()  # Newline at the end


if __name__ == "__main__":
    asyncio.run(main())
