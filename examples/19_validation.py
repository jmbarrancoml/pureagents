"""Output validation - validate and retry responses."""

import asyncio
import json

from pure_agents import Agent


def is_valid_json(response: str) -> bool:
    """Validate that response is valid JSON."""
    try:
        json.loads(response)
        return True
    except json.JSONDecodeError:
        return False


def contains_keywords(response: str) -> bool:
    """Validate that response contains required keywords."""
    required = ["python", "programming"]
    return any(kw in response.lower() for kw in required)


def min_length(min_chars: int):
    """Create a validator for minimum length."""

    def validator(response: str) -> bool:
        return len(response) >= min_chars

    return validator


async def main():
    # JSON validation with retry
    print("=== JSON Validation ===")
    agent_json = Agent(
        validator=is_valid_json,
        validation_retries=2,
        debug=True,
    )
    result = await agent_json.run(
        "Return a JSON object with name and age fields. "
        "Only return the JSON, no other text."
    )
    print(f"Result: {result}")

    # Keyword validation
    print("\n=== Keyword Validation ===")
    agent_keywords = Agent(
        validator=contains_keywords,
        validation_retries=2,
    )
    result = await agent_keywords.run("Tell me about a popular programming language")
    print(f"Result: {result[:100]}...")

    # Minimum length validation
    print("\n=== Length Validation ===")
    agent_length = Agent(
        validator=min_length(200),
        validation_retries=1,
    )
    result = await agent_length.run("Explain what an API is in detail")
    print(f"Result length: {len(result)} characters")


if __name__ == "__main__":
    asyncio.run(main())
