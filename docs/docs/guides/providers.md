---
sidebar_position: 2
---

# Providers

pureagents supports multiple LLM providers.

## Available providers

| Provider | Default model | Environment variable |
|----------|---------------|---------------------|
| `mistral` | `mistral-large-latest` | `MISTRAL_API_KEY` |
| `openai` | `gpt-5.6-luna` | `OPENAI_API_KEY` |
| `anthropic` | `claude-sonnet-5` | `ANTHROPIC_API_KEY` |

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

# GPT-5.6 tiers, cheapest first
agent = Agent(provider="openai", model="gpt-5.6-luna")  # Default
agent = Agent(provider="openai", model="gpt-5.6-terra")  # Balanced
agent = Agent(provider="openai", model="gpt-5.6-sol")  # Highest capability
```

## Anthropic

```python
agent = Agent(provider="anthropic")

# Claude tiers, cheapest first
agent = Agent(provider="anthropic", model="claude-haiku-4-5")  # Fastest
agent = Agent(provider="anthropic", model="claude-sonnet-5")  # Default
agent = Agent(provider="anthropic", model="claude-opus-5")  # Agentic coding
agent = Agent(provider="anthropic", model="claude-fable-5")  # Long-running agents
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
