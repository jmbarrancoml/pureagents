"""Tool groups - enable/disable sets of tools dynamically."""

import asyncio

from pure_agents import Agent, tool


# Web tools
@tool(group="web")
def search(query: str) -> str:
    """Search the web."""
    return f"Web results for {query}"


@tool(group="web")
def fetch_url(url: str) -> str:
    """Fetch content from a URL."""
    return f"Content from {url}"


# Math tools
@tool(group="math")
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    return str(eval(expression))


@tool(group="math")
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between units."""
    return f"{value} {from_unit} = {value * 2} {to_unit}"


# No group - always available
@tool
def get_time() -> str:
    """Get current time."""
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")


async def main():
    # All tools enabled by default
    print("=== All Tools ===")
    agent = Agent(tools=[search, fetch_url, calculate, convert_units, get_time])
    print(f"Available tools: {list(agent.tools.keys())}")

    # Only web tools
    print("\n=== Only Web Tools ===")
    agent_web = Agent(
        tools=[search, fetch_url, calculate, convert_units, get_time],
        enabled_groups=["web"],
    )
    print(f"Available tools: {list(agent_web.tools.keys())}")

    # Only math tools
    print("\n=== Only Math Tools ===")
    agent_math = Agent(
        tools=[search, fetch_url, calculate, convert_units, get_time],
        enabled_groups=["math"],
    )
    print(f"Available tools: {list(agent_math.tools.keys())}")

    # Enable/disable dynamically
    print("\n=== Dynamic Groups ===")
    agent.enabled_groups = []
    print(f"No groups enabled: {list(agent.tools.keys())}")

    agent.enable_group("web")
    print(f"Web enabled: {list(agent.tools.keys())}")

    agent.enable_group("math")
    print(f"Web + Math enabled: {list(agent.tools.keys())}")

    agent.disable_group("web")
    print(f"Only Math: {list(agent.tools.keys())}")


if __name__ == "__main__":
    asyncio.run(main())
