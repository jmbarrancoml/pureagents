---
sidebar_position: 2
---

# Providers

pureagents supports multiple LLM providers.

## Available providers

| Provider | Default model | Environment variable |
|----------|---------------|---------------------|
| `mistral` | `mistral-large-latest` | `MISTRAL_API_KEY` |
| `openai` | `gpt-5.2-instant` | `OPENAI_API_KEY` |
| `anthropic` | `claude-opus-5` | `ANTHROPIC_API_KEY` |

## Mistral (default)

```python
from pure_agents import Agent

# Uses MISTRAL_API_KEY from environment
agent = Agent()

# Or explicit
agent = Agent(api_key="your-mistral-key")

# Specific model
agent = Agent(model="mistral-small-latest")
```

## OpenAI

```python
agent = Agent(provider="openai")

# GPT-5.2 variants
agent = Agent(provider="openai", model="gpt-5.2-instant")  # Fast
agent = Agent(provider="openai", model="gpt-5.2-thinking")  # Reasoning
agent = Agent(provider="openai", model="gpt-5.2-pro")  # Best quality
agent = Agent(provider="openai", model="gpt-5.2-codex")  # Code
```

## Anthropic

```python
agent = Agent(provider="anthropic")

# Claude variants
agent = Agent(provider="anthropic", model="claude-opus-5")
agent = Agent(provider="anthropic", model="claude-opus-4-5-20251101")
```

## Explicit API key

You can pass the API key directly:

```python
agent = Agent(provider="openai", api_key="sk-...")
```

:::warning
Don't commit API keys to your repository. Use environment variables.
:::

## Custom base URL

For self-hosted or proxy endpoints, modify the client directly:

```python
from pure_agents.clients import LLMClient

client = LLMClient(api_key="your-key", base_url="https://your-proxy.com/v1")
```
