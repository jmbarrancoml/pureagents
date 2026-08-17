---
sidebar_position: 12
---

# Images

Use vision models to analyse images.

## Basic usage

```python
agent = Agent(provider="openai")  # Use a vision-capable model

result = await agent.run(
    "What's in this image?",
    images=["photo.jpg"],
)
```

## Multiple images

```python
result = await agent.run(
    "Compare these two images",
    images=["image1.png", "image2.png"],
)
```

## Supported formats

- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- GIF (`.gif`)
- WebP (`.webp`)

## URL images

Pass URLs directly:

```python
result = await agent.run(
    "Describe this image",
    images=["https://example.com/photo.jpg"],
)
```

## With tools

Combine vision with tools:

```python
@tool
def save_description(text: str) -> str:
    """Save image description to file."""
    with open("description.txt", "w") as f:
        f.write(text)
    return "Saved!"


agent = Agent(provider="openai", tools=[save_description])

result = await agent.run(
    "Describe this image and save the description",
    images=["photo.jpg"],
)
```

## Provider support

| Provider | Vision support |
|----------|----------------|
| OpenAI | ✅ GPT-4 Vision |
| Anthropic | ✅ Claude 3 |
| Mistral | ✅ Pixtral |

## How it works

Images are:
1. Loaded from disk or URL
2. Base64 encoded
3. Sent with the appropriate MIME type
4. Included in the message content

The encoding is handled automatically. Just pass file paths or URLs.
