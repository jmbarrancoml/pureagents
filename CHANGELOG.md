# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed (breaking)

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
- Default models are now `mistral-large-latest`, `gpt-5.2-instant` and
  `claude-opus-5`.

### Added

- `Agent(max_tokens=)`, `fallback_model=` and `fallback_api_key=`.
- `Agent.aclose()` and async context manager support.
- `batch(prompts, max_concurrency=5)`.
- `set_cache_size()`.
- `Usage.set_rates()`.
- `Graph` uses the first node added as its entry point.

### Fixed

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
