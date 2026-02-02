"""Demo of pureagents."""

from pure_agents import Agent, tool


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    return str(eval(expression))


@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


async def main():
    agent = Agent(tools=[calculate, greet], debug=True)

    # Non-streaming
    print("=== Non-streaming ===")
    result = await agent.run("What is 25 * 4?")
    print(f"Result: {result}")

    # Streaming
    print("\n=== Streaming ===")
    async for chunk in agent.stream("Explain why 25 * 4 equals 100"):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
