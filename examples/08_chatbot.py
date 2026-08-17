"""Simple chatbot - interactive CLI conversation."""

import asyncio

from pure_agents import Agent, tool


@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    # Mock data - replace with real API
    weather_data = {
        "madrid": "22°C, sunny",
        "london": "15°C, cloudy",
        "paris": "18°C, partly cloudy",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
def set_reminder(message: str, time: str) -> str:
    """Set a reminder for a specific time."""
    return f"Reminder set: '{message}' at {time}"


async def main():
    print("=== pureagents Chatbot ===")
    print("Type 'quit' to exit\n")

    # Create agent with session for persistent memory
    agent = Agent(
        tools=[get_weather, set_reminder],
        session="chatbot-demo",
        system=(
            "You are a helpful assistant. You can check weather and set reminders. "
            "Be friendly and concise."
        ),
    )

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Stream the response
            print("Bot: ", end="", flush=True)
            async for event in agent.stream(user_input):
                if event.type == "text":
                    print(event.content, end="", flush=True)
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    asyncio.run(main())
