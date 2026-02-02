"""Batch processing - run multiple prompts in parallel."""

import asyncio
import time

from pure_agents import Agent


async def main():
    agent = Agent()

    prompts = [
        "What is the capital of France?",
        "What is 2 + 2?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
        "Name three programming languages.",
    ]

    # Run sequentially (slow)
    print("=== Sequential Execution ===")
    start = time.time()
    sequential_results = []
    for prompt in prompts:
        fresh_agent = Agent()
        result = await fresh_agent.run(prompt)
        sequential_results.append(result)
    sequential_time = time.time() - start
    print(f"Time: {sequential_time:.2f}s")

    # Run in parallel with batch (fast)
    print("\n=== Parallel Execution (batch) ===")
    start = time.time()
    parallel_results = await agent.batch(prompts)
    parallel_time = time.time() - start
    print(f"Time: {parallel_time:.2f}s")

    # Show results
    print("\n=== Results ===")
    for prompt, result in zip(prompts, parallel_results):
        print(f"Q: {prompt}")
        print(f"A: {result[:100]}...")
        print()

    # Speedup
    print(f"Speedup: {sequential_time / parallel_time:.1f}x faster")


if __name__ == "__main__":
    asyncio.run(main())
