"""Image support - send images to vision models."""

from pure_agents import Agent


async def main():
    # Use a vision-capable model
    agent = Agent(
        provider="openai",
        model="gpt-5.2-instant",  # Vision-capable model
    )

    # Describe an image
    result = await agent.run(
        "What do you see in this image?",
        images=["photo.jpg"],  # Path to image file
    )
    print(result)

    # Multiple images
    result = await agent.run(
        "Compare these two images",
        images=["image1.png", "image2.png"],
    )
    print(result)

    # With Anthropic
    agent_claude = Agent(
        provider="anthropic",
        model="claude-opus-5",
    )

    result = await agent_claude.run(
        "Describe this diagram",
        images=["diagram.png"],
    )
    print(result)


# Note: This example requires actual image files to run
if __name__ == "__main__":
    print("This example requires image files.")
    print("Usage: agent.run('Describe this', images=['photo.jpg'])")
