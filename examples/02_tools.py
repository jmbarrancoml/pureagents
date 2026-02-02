"""Using tools - give the agent capabilities."""

import asyncio
from datetime import datetime

from pure_agents import Agent, tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In real code, call a weather API here
    return f"Weather in {city}: 22°C, sunny"


@tool
def get_time(timezone: str) -> str:
    """Get the current time in a timezone."""
    # Simplified - in real code use pytz
    return f"Current time: {datetime.now().strftime('%H:%M')}"


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)  # Be careful with eval in production!
        return str(result)
    except Exception as e:
        return f"Error: {e}"


async def main():
    agent = Agent(
        tools=[get_weather, get_time, calculate],
        debug=True,  # See tool calls in action
    )

    # The agent will use tools as needed
    response = await agent.run(
        "What's the weather in Madrid and what's 15% of 340?"
    )
    print(f"\nFinal: {response}")


if __name__ == "__main__":
    asyncio.run(main())
