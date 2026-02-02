"""Parallel tool execution - multiple tools run concurrently."""

import asyncio
import time

from pure_agents import Agent, tool


@tool
async def fetch_weather(city: str) -> str:
    """Fetch weather for a city (simulated delay)."""
    await asyncio.sleep(1)  # Simulate API call
    return f"Weather in {city}: 22°C, sunny"


@tool
async def fetch_news(topic: str) -> str:
    """Fetch news about a topic (simulated delay)."""
    await asyncio.sleep(1)  # Simulate API call
    return f"News about {topic}: 5 articles found"


@tool
async def fetch_stock(symbol: str) -> str:
    """Fetch stock price (simulated delay)."""
    await asyncio.sleep(1)  # Simulate API call
    return f"Stock {symbol}: $150.25"


async def main():
    agent = Agent(
        tools=[fetch_weather, fetch_news, fetch_stock],
        debug=True,
    )

    # When the model returns multiple tool calls, they run in parallel
    print("=== Parallel Tool Execution ===")
    print("Each tool takes 1 second, but parallel execution is faster\n")

    start = time.time()
    result = await agent.run(
        "Get me the weather in Madrid, news about AI, and the stock price for AAPL. "
        "I need all three pieces of information."
    )
    elapsed = time.time() - start

    print(f"\nResult: {result}")
    print(f"\nTotal time: {elapsed:.2f}s")
    print("(Would be ~3s sequential, but parallel is faster)")


if __name__ == "__main__":
    asyncio.run(main())
