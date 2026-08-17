"""
Routing: direct prompts to specialised agents.

Two approaches:
1. Function-based: you define the routing logic
2. LLM-based: the router decides automatically
"""

import asyncio

from pure_agents import Agent, Router


async def main():
    # Create specialised agents
    coder = Agent(
        template="coder",
        system="You are an expert programmer. Write clean, working code.",
    )

    writer = Agent(
        template="creative",
        system="You are a creative writer. Write engaging, vivid prose.",
    )

    analyst = Agent(
        template="analyst",
        system="You are a data analyst. Provide clear, logical analysis.",
    )

    # --- Option 1: Function-based routing ---
    CODE_WORDS = ["code", "function", "program", "script"]
    CREATIVE_WORDS = ["write", "story", "poem", "creative"]

    def route_fn(prompt: str) -> str:
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in CODE_WORDS):
            return "coder"
        if any(word in prompt_lower for word in CREATIVE_WORDS):
            return "writer"
        return "analyst"

    router = Router(
        agents={"coder": coder, "writer": writer, "analyst": analyst},
        route=route_fn,
    )

    print("=== Function-based routing ===\n")

    result = await router.run("Write a Python function to calculate fibonacci")
    print(f"Code request:\n{result}\n")

    result = await router.run("Write a short poem about the sea")
    print(f"Creative request:\n{result}\n")

    # --- Option 2: LLM-based routing ---
    llm_router = Router(
        agents={"coder": coder, "writer": writer, "analyst": analyst},
        # No route function = LLM decides
    )

    print("=== LLM-based routing ===\n")

    result = await llm_router.run("Explain the trade-offs between SQL and NoSQL")
    print(f"Analysis request:\n{result}\n")


if __name__ == "__main__":
    asyncio.run(main())
