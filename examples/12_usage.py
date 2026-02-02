"""Usage tracking - monitor tokens and estimate costs."""

import asyncio

from pure_agents import Agent


async def main():
    agent = Agent()

    # Run some prompts
    print("=== Running Prompts ===")

    await agent.run("What is Python?")
    print(f"After 1st prompt: {agent.usage.total_tokens} tokens")

    await agent.run("What are its main features?")
    print(f"After 2nd prompt: {agent.usage.total_tokens} tokens")

    await agent.run("Give me a code example")
    print(f"After 3rd prompt: {agent.usage.total_tokens} tokens")

    # Show usage summary
    print("\n=== Usage Summary ===")
    print(f"Requests: {agent.usage.requests}")
    print(f"Input tokens: {agent.usage.input_tokens}")
    print(f"Output tokens: {agent.usage.output_tokens}")
    print(f"Total tokens: {agent.usage.total_tokens}")

    # Estimate cost
    print("\n=== Cost Estimate ===")
    print(f"Mistral: ${agent.usage.cost('mistral'):.4f}")
    print(f"OpenAI: ${agent.usage.cost('openai'):.4f}")
    print(f"Anthropic: ${agent.usage.cost('anthropic'):.4f}")

    # Reset usage
    agent.usage.reset()
    print(f"\nAfter reset: {agent.usage.total_tokens} tokens")


if __name__ == "__main__":
    asyncio.run(main())
