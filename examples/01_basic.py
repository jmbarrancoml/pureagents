"""Basic agent usage - simple question answering."""

import asyncio

from pure_agents import Agent


async def main():
    # Create an agent (uses Mistral by default)
    agent = Agent()

    # Ask a question
    response = await agent.run("What are the three laws of robotics?")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
