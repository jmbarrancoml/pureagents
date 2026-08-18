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

## Any OpenAI-compatible endpoint

Most of the ecosystem speaks the OpenAI wire format. Point `base_url` at it and
you are done:

```python
agent = Agent(base_url="http://localhost:11434/v1", model="llama3.3")
```

No API key is needed when the server does not check one. Pass `api_key=` when
it does.

### Common endpoints

| Runtime | `base_url` | Key |
|---------|-----------|-----|
| Ollama | `http://localhost:11434/v1` | none |
| LM Studio | `http://localhost:1234/v1` | none |
| vLLM | `http://localhost:8000/v1` | none |
| OpenRouter | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| Groq | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` |
| Together | `https://api.together.xyz/v1` | `TOGETHER_API_KEY` |

Model names are yours to supply. pureagents does not ship defaults for these
because their catalogues move faster than a release cycle.

### Registering one by name

If you use an endpoint more than once, give it a name:

```python
from pure_agents import Agent, register_provider

register_provider("ollama", base_url="http://localhost:11434/v1")

agent = Agent(provider="ollama", model="llama3.3")
```

`register_provider` takes:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_url` | Required | Endpoint root, without a trailing slash |
| `env_var` | `None` | Environment variable holding the key. Omit for a keyless local server |
| `default_model` | `None` | Used when the caller omits `model=`. Omit and callers must be explicit |
| `dialect` | `"openai"` | `"openai"` or `"anthropic"` wire format |
| `headers` | `None` | Extra headers every request needs |
| `overwrite` | `False` | Allow replacing an existing name |

Built-in providers are protected: registering over `openai`, `mistral` or
`anthropic` needs `overwrite=True`, and `unregister_provider` refuses them
outright.

### Extra headers

Some gateways want attribution headers:

```python
agent = Agent(
    base_url="https://openrouter.ai/api/v1",
    model="anthropic/claude-sonnet-5",
    headers={"HTTP-Referer": "https://myapp.dev", "X-Title": "My App"},
)
```

Headers passed to `Agent` win over headers set on the registered provider.

### Local models for development

Running the loop against a local model costs nothing, which makes it a good
default while you iterate on prompts and tools:

```python
import os

if os.environ.get("DEV"):
    agent = Agent(base_url="http://localhost:11434/v1", model="llama3.3")
else:
    agent = Agent(provider="anthropic")
```

Tool calling and streaming both work against Ollama, LM Studio and vLLM.

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
