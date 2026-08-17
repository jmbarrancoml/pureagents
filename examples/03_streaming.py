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

    # Stream the run as typed events
    async for event in agent.stream("Tell me about the history of Python"):
        if event.type == "text":
            print(event.content, end="", flush=True)
        elif event.type == "tool_call":
            print(f"\n[calling {event.name} with {event.arguments}]")
        elif event.type == "tool_result":
            print(f"[{event.name} returned: {event.content}]\n", end="", flush=True)
        elif event.type == "done":
            print()  # Newline at the end


if __name__ == "__main__":
    asyncio.run(main())
