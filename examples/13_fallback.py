"""Fallback providers - automatic failover when primary fails."""

import asyncio

from pure_agents import Agent


async def main():
    # If OpenAI fails, automatically try Mistral
    agent = Agent(
        provider="openai",
        fallback="mistral",
        debug=True,
    )

    # The agent will try OpenAI first, then Mistral if it fails
    result = await agent.run("What is the capital of Spain?")
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
