"""Response caching - save money on repeated queries."""

import asyncio
import time

from pure_agents import Agent, clear_cache


async def main():
    # Enable caching
    agent = Agent(cache=True, debug=True)

    prompt = "What is 2 + 2?"

    # First call - hits the API
    print("=== First Call (API) ===")
    start = time.time()
    result1 = await agent.run(prompt)
    print(f"Result: {result1}")
    print(f"Time: {time.time() - start:.2f}s")

    # Second call - hits the cache
    print("\n=== Second Call (Cache) ===")
    start = time.time()
    result2 = await agent.run(prompt)
    print(f"Result: {result2}")
    print(f"Time: {time.time() - start:.4f}s")  # Much faster!

    # Clear cache if needed
    print("\n=== Clear Cache ===")
    clear_cache()
    print("Cache cleared")

    # Third call - hits the API again
    print("\n=== Third Call (API) ===")
    start = time.time()
    result3 = await agent.run(prompt)
    print(f"Result: {result3}")
    print(f"Time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
