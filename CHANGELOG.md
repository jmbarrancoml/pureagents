# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `output=` is now enforced by the provider while it decodes, instead of being
  requested in the prompt. OpenAI-compatible endpoints get
  `response_format.json_schema` with `strict: true`; Anthropic gets
  `output_config.format`. Schemas are tightened for strict mode automatically:
  objects forbid extra properties, every property is required, and a field with
  a default becomes nullable.
- `register_provider(structured_outputs=False)` falls back to the old
  prompt-based method for an endpoint that rejects `response_format`.
- `register_provider(strict_schemas=True)` opts an endpoint into OpenAI's
  strict mode. Built-in `openai` and `mistral` have it on; it is off
  elsewhere, since it is an OpenAI extension rather than part of the wire
  format every compatible server implements.

- Any OpenAI-compatible endpoint works: `Agent(base_url="http://localhost:11434/v1",
  model="llama3.3")` covers Ollama, LM Studio, vLLM, OpenRouter, Groq, Together
  and self-hosted gateways. No key is required when the server does not ask for
  one.
- `register_provider()` names an endpoint for reuse, with `env_var`,
  `default_model`, `dialect` (`"openai"` or `"anthropic"`) and `headers`.
  `unregister_provider()` removes it; built-ins are protected.
- `Agent(headers=...)` adds request headers, for gateways that want attribution.

### Changed (breaking)

- `PROVIDERS` maps names to a frozen `Provider` record rather than a plain dict.
  Read `PROVIDERS["openai"].base_url` instead of `PROVIDERS["openai"]["base_url"]`.
- `Agent(provider=)` defaults to `None` and resolves to `"mistral"`, so that
  passing `base_url=` and `provider=` together can be rejected.

- `Agent(system_prompt=...)` is now `Agent(system=...)`, and the attribute is
  `agent.system`.
- `agent.stream()` yields `StreamEvent` objects instead of plain strings. Each
  has `type` (`text`, `tool_call`, `tool_result`, `done`), `content`, `name`,
  `id` and `arguments`.
- Running out of steps raises `MaxStepsError` instead of returning the string
  "Max steps reached without final answer."
- A reply that will not parse into the requested dataclass raises
  `StructuredOutputError` instead of a bare `JSONDecodeError` or `TypeError`.
- `Usage.cost()` takes a model, not a provider, and returns `None` when the
  model has no published rate on file. Use `Usage.set_rates()` for your own.
- Session ids are validated: letters, digits, dots, dashes and underscores.
- Default models are now `mistral-large-latest`, `gpt-5.6-luna` and
  `claude-sonnet-5`: the cheapest current-generation tier each provider
  offers that still drives a tool loop reliably.

### Added

- `Agent(max_tokens=)`, `fallback_model=` and `fallback_api_key=`.
- `Agent.aclose()` and async context manager support.
- `batch(prompts, max_concurrency=5)`.
- `set_cache_size()`.
- `Usage.set_rates()`.
- `Graph` uses the first node added as its entry point.

### Fixed

- A reply cut off at the token limit now raises
  `TruncatedResponseError` on OpenAI-compatible endpoints too. Only the
  Anthropic client checked for it, so elsewhere a truncated tool call or a
  half-written object surfaced as "did not return valid JSON", blaming the
  model for the caller's token limit.
- An explicit `api_key=` no longer loses to the environment variable.
- The fallback provider uses its own API key and its own model instead of
  sending the primary provider's model to a different API.
- The response cache keys on the system prompt, provider and tool set, is
  bounded, keeps the conversation consistent on a hit, and honours `output=`.
- A malformed tool-call argument string is reported back to the model rather
  than crashing the run, and one failing call no longer cancels its siblings.
- `max_messages` no longer separates a tool result from its tool call.
- `@tool(timeout=)` works on sync functions, and parallel tool calls really run
  in parallel.
- Type hints become correct JSON Schema: typed lists and dicts, `Optional`,
  `Literal`, `Enum` and nested dataclasses. Only parameters without defaults
  are marked required.
- Parallel tool calls against Anthropic no longer produce consecutive user
  messages.
- `Agent(timeout=)` reaches httpx instead of being capped at 60 seconds, and
  HTTP connections are pooled instead of rebuilt per request.
- Retries cover 429, 5xx and network failures only, honour `Retry-After`, and
  are jittered.
- Streaming goes through retries, timeout, fallback and token accounting, and
  supports images.
- `batch()` clones the full agent configuration and bounds concurrency.
- Session files are written atomically.

## [0.1.0] - 2026-02-02

### Added

- Agent class with tool support
- `@tool` decorator for creating tools from functions
- Mistral, OpenAI, and Anthropic providers
- Streaming support
- Session-based memory persistence
- Structured outputs with dataclasses
- Hooks for monitoring (`on_tool_call`, `on_tool_result`, `on_thinking`, `on_plan`)
- Batch execution
- Retry with exponential backoff
- Timeout support (agent and per-tool)
- Context limit management
- Usage tracking (tokens and costs)
- Tool groups for dynamic enable/disable
- Image support for vision models
- Chaining: sequential agent execution
- Routing: dynamic agent selection
- Planning: think-before-acting mode
- Graph: multi-agent workflows with conditional edges

[Unreleased]: https://github.com/jmbarrancoml/pureagents/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jmbarrancoml/pureagents/releases/tag/v0.1.0
