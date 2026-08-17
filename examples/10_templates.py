"""Templates - predefined system prompts for common use cases."""

import asyncio

from pure_agents import TEMPLATES, Agent


async def main():
    # See available templates
    print("=== Available Templates ===")
    for name, prompt in TEMPLATES.items():
        print(f"  {name}: {prompt[:50]}...")

    # Use the coder template
    print("\n=== Coder Template ===")
    coder = Agent(template="coder")
    response = await coder.run("Write a Python function to reverse a string")
    print(response)

    # Use the researcher template
    print("\n=== Researcher Template ===")
    researcher = Agent(template="researcher")
    response = await researcher.run("What are the main causes of climate change?")
    print(response)

    # Use the creative template
    print("\n=== Creative Template ===")
    creative = Agent(template="creative")
    response = await creative.run("Write a haiku about programming")
    print(response)

    # Use the tutor template
    print("\n=== Tutor Template ===")
    tutor = Agent(template="tutor")
    response = await tutor.run("Explain recursion to a beginner")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
