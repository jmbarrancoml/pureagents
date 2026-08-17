"""Multiple providers - switch between LLMs easily."""

import asyncio
import os

from pure_agents import Agent


async def main():
    prompt = "Explain quantum computing in one sentence."

    # Mistral (default) - requires MISTRAL_API_KEY
    if os.environ.get("MISTRAL_API_KEY"):
        print("=== Mistral ===")
        agent = Agent(provider="mistral")
        response = await agent.run(prompt)
        print(f"{response}\n")

    # OpenAI - requires OPENAI_API_KEY
    if os.environ.get("OPENAI_API_KEY"):
        print("=== OpenAI ===")
        agent = Agent(provider="openai")
        response = await agent.run(prompt)
        print(f"{response}\n")

    # Anthropic - requires ANTHROPIC_API_KEY
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("=== Anthropic ===")
        agent = Agent(provider="anthropic")
        response = await agent.run(prompt)
        print(f"{response}\n")

    # Use a specific model
    if os.environ.get("OPENAI_API_KEY"):
        print("=== OpenAI (specific model) ===")
        agent = Agent(provider="openai", model="gpt-5.2-instant")
        response = await agent.run(prompt)
        print(f"{response}\n")


if __name__ == "__main__":
    asyncio.run(main())
