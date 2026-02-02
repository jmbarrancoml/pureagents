"""Tool choice - control when tools are used."""

import asyncio

from pure_agents import Agent, tool


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    return str(eval(expression))


@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for {query}"


async def main():
    # Auto (default) - model decides when to use tools
    print("=== Auto (default) ===")
    agent_auto = Agent(tools=[calculator, search], tool_choice="auto")
    result = await agent_auto.run("What is 15 * 7?")
    print(f"Result: {result}")

    # Required - force the model to use a tool
    print("\n=== Required ===")
    agent_required = Agent(tools=[calculator], tool_choice="required", debug=True)
    result = await agent_required.run("Calculate 100 / 4")
    print(f"Result: {result}")

    # None - prevent tool use
    print("\n=== None (no tools) ===")
    agent_none = Agent(tools=[calculator], tool_choice="none")
    result = await agent_none.run("What is 5 + 5? Don't use any tools.")
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
