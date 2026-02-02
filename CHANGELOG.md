# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
