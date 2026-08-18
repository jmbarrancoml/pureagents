"""Any OpenAI-compatible endpoint: local models, gateways, your own server."""

import asyncio

from pure_agents import Agent, register_provider, tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


async def main():
    # === Ad-hoc: point at a local server ===
    print("=== Ollama ===")
    local = Agent(
        base_url="http://localhost:11434/v1",
        model="llama3.3",
        tools=[add],
    )
    print(await local.run("What is 21 + 21?"))

    # === Named, for reuse across the codebase ===
    print("\n=== Registered provider ===")
    register_provider("ollama", base_url="http://localhost:11434/v1")
    same_server = Agent(provider="ollama", model="llama3.3")
    print(await same_server.run("Say hi in one word."))

    # === A gateway that wants attribution headers ===
    print("\n=== OpenRouter ===")
    register_provider(
        "openrouter",
        base_url="https://openrouter.ai/api/v1",
        env_var="OPENROUTER_API_KEY",
        headers={"HTTP-Referer": "https://myapp.dev", "X-Title": "My App"},
    )
    routed = Agent(provider="openrouter", model="anthropic/claude-sonnet-5")
    print(await routed.run("Say hi in one word."))

    # === Local while developing, hosted in production ===
    print("\n=== Swap by environment ===")
    import os

    if os.environ.get("DEV"):
        agent = Agent(base_url="http://localhost:11434/v1", model="llama3.3")
    else:
        agent = Agent(provider="anthropic")
    print(f"Using {agent.provider} / {agent.model}")


if __name__ == "__main__":
    asyncio.run(main())
