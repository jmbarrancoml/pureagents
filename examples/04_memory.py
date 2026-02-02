"""Memory and sessions - persistent conversations."""

import asyncio

from pure_agents import Agent


async def main():
    # Create an agent with a session ID - conversations persist automatically
    agent = Agent(session="my-assistant")

    # First conversation
    print("You: My name is Carlos and I like pizza")
    response = await agent.run("My name is Carlos and I like pizza")
    print(f"Agent: {response}\n")

    # Continue the conversation - agent remembers context
    print("You: What's my name and what do I like?")
    response = await agent.run("What's my name and what do I like?")
    print(f"Agent: {response}\n")

    # Even in a new agent instance with the same session, memory persists
    agent2 = Agent(session="my-assistant")
    print("You: (new instance) Remind me, what's my favourite food?")
    response = await agent2.run("Remind me, what's my favourite food?")
    print(f"Agent: {response}\n")

    # Clear memory when done
    agent2.clear()
    print("Session cleared.")


if __name__ == "__main__":
    asyncio.run(main())
