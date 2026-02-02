"""
Planning: agent creates a plan before executing.

With plan=True, the agent:
1. Creates a step-by-step plan
2. Executes the plan using tools as needed
"""

import asyncio

from pure_agents import Agent, tool


@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': [relevant information here]"


@tool
def write_file(filename: str, content: str) -> str:
    """Write content to a file."""
    print(f"  Writing to {filename}...")
    return f"Wrote {len(content)} characters to {filename}"


@tool
def read_file(filename: str) -> str:
    """Read content from a file."""
    return f"Contents of {filename}: [file contents here]"


async def main():
    agent = Agent(
        tools=[search, write_file, read_file],
        debug=True,
        on_plan=lambda p: print(f"\n=== PLAN ===\n{p}\n=== END PLAN ===\n"),
    )

    # Without planning
    print("=== Without planning ===\n")
    result = await agent.run("What is the capital of France?")
    print(f"Result: {result}\n")

    # With planning - agent creates plan first, then executes
    agent.clear()
    print("\n=== With planning ===\n")
    result = await agent.run(
        "Research the history of Python programming language and write a summary",
        plan=True,
    )
    print(f"\nFinal result:\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
