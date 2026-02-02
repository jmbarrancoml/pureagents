"""Structured outputs - get typed responses."""

import asyncio
from dataclasses import dataclass

from pure_agents import Agent


@dataclass
class SentimentAnalysis:
    sentiment: str  # "positive", "negative", "neutral"
    confidence: float  # 0.0 to 1.0
    summary: str


@dataclass
class Recipe:
    name: str
    ingredients: str  # comma-separated
    steps: str
    prep_time_minutes: int


async def main():
    agent = Agent()

    # Sentiment analysis with structured output
    print("=== Sentiment Analysis ===")
    text = "I absolutely love this new phone! The camera is incredible."

    result = await agent.run(
        f"Analyse the sentiment of this text: '{text}'",
        output=SentimentAnalysis,
    )

    print(f"Sentiment: {result.sentiment}")
    print(f"Confidence: {result.confidence}")
    print(f"Summary: {result.summary}")

    # Recipe extraction
    print("\n=== Recipe Extraction ===")
    agent2 = Agent()  # Fresh agent

    recipe = await agent2.run(
        "Give me a simple recipe for scrambled eggs",
        output=Recipe,
    )

    print(f"Recipe: {recipe.name}")
    print(f"Ingredients: {recipe.ingredients}")
    print(f"Steps: {recipe.steps}")
    print(f"Prep time: {recipe.prep_time_minutes} minutes")


if __name__ == "__main__":
    asyncio.run(main())
