"""
Chaining: run agents in sequence.

Output of each agent feeds into the next.
Useful for multi-step workflows like research → summarise → format.
"""

import asyncio

from pure_agents import Agent, chain


async def main():
    # Agent 1: Research and gather information
    researcher = Agent(
        template="researcher",
        system_prompt=(
            "You are a researcher. Given a topic, provide detailed factual information. "
            "Be thorough and include specific details."
        ),
    )

    # Agent 2: Summarise and simplify
    summariser = Agent(
        system_prompt=(
            "You are a summariser. Given a detailed text, create a concise summary. "
            "Keep only the most important points. Use bullet points."
        ),
    )

    # Agent 3: Format for social media
    formatter = Agent(
        system_prompt=(
            "You are a social media writer. Given content, rewrite it as an engaging "
            "Twitter thread. Use emojis sparingly. Keep each tweet under 280 chars."
        ),
    )

    # Run the chain
    result = await chain(
        [researcher, summariser, formatter],
        "The history of the Python programming language",
    )

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
