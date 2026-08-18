"""Structured outputs - get typed responses.

The schema is sent to the provider as a decoding constraint, so the reply
cannot come back the wrong shape.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from pure_agents import Agent, StructuredOutputError


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class SentimentAnalysis:
    sentiment: Sentiment
    confidence: float  # 0.0 to 1.0
    summary: str
    keywords: list[str]


@dataclass
class Ingredient:
    name: str
    quantity: str


@dataclass
class Recipe:
    name: str
    ingredients: list[Ingredient]
    steps: list[str]
    prep_time_minutes: int
    difficulty: Literal["easy", "medium", "hard"]
    notes: str | None = None
    tags: list[str] = field(default_factory=list)


async def main():
    agent = Agent()

    # === Enums and lists ===
    print("=== Sentiment Analysis ===")
    text = "I absolutely love this new phone! The camera is incredible."

    result = await agent.run(
        f"Analyse the sentiment of this text: '{text}'",
        output=SentimentAnalysis,
    )

    print(f"Sentiment: {result.sentiment}")
    print(f"Confidence: {result.confidence}")
    print(f"Keywords: {', '.join(result.keywords)}")

    # === Nested dataclasses and optional fields ===
    print("\n=== Recipe Extraction ===")
    agent.clear()

    recipe = await agent.run(
        "Give me a simple recipe for scrambled eggs",
        output=Recipe,
    )

    print(f"{recipe.name} ({recipe.difficulty}, {recipe.prep_time_minutes} min)")
    for ingredient in recipe.ingredients:
        print(f"  - {ingredient.quantity} {ingredient.name}")
    for index, step in enumerate(recipe.steps, 1):
        print(f"  {index}. {step}")
    if recipe.notes:
        print(f"Notes: {recipe.notes}")

    # === What a failure looks like ===
    print("\n=== Error handling ===")
    agent.clear()
    try:
        await agent.run("Write a poem about the sea", output=Recipe)
    except StructuredOutputError as e:
        print(f"Did not fit the shape. Model said: {e.raw[:80]}...")


if __name__ == "__main__":
    asyncio.run(main())
