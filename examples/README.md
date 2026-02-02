# pureagents examples

Practical examples demonstrating pureagents features.

## Setup

```bash
pip install pureagents

export MISTRAL_API_KEY="your-key"  # Default provider
export OPENAI_API_KEY="your-key"   # For OpenAI/images
export ANTHROPIC_API_KEY="your-key"
```

## Examples

### Basics
| File | Description |
|------|-------------|
| `01_basic.py` | Simple question answering |
| `02_tools.py` | Using tools (weather, calculator) |
| `03_streaming.py` | Real-time streaming responses |
| `04_memory.py` | Persistent conversations |
| `05_structured_output.py` | Typed dataclass responses |
| `06_hooks.py` | Monitor with callbacks |
| `07_providers.py` | Mistral, OpenAI, Anthropic |
| `08_chatbot.py` | Interactive CLI chatbot |

### Reliability
| File | Description |
|------|-------------|
| `09_reliability.py` | Retry, timeout, context limits |
| `13_fallback.py` | Automatic provider failover |
| `14_tool_timeout.py` | Per-tool timeout limits |
| `15_caching.py` | Response caching |

### Advanced
| File | Description |
|------|-------------|
| `10_templates.py` | Predefined system prompts |
| `11_batch.py` | Parallel prompt execution |
| `12_usage.py` | Token and cost tracking |
| `16_tool_choice.py` | Force/prevent tool use |
| `17_tool_groups.py` | Enable/disable tool sets |
| `18_images.py` | Vision model support |
| `19_validation.py` | Output validation with retry |
| `20_parallel_tools.py` | Concurrent tool execution |

### Orchestration
| File | Description |
|------|-------------|
| `21_chaining.py` | Sequential agent pipelines |
| `22_routing.py` | Dynamic agent selection |
| `23_planning.py` | Think-before-acting mode |
| `24_graph.py` | Multi-agent workflows with conditional edges |

## Running

```bash
python examples/01_basic.py
```
