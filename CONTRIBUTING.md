# Contributing to pureagents

Thanks for your interest in contributing.

## Philosophy

pureagents is intentionally minimal. Before adding features, consider:

- Does this add complexity?
- Can this be achieved with existing features?
- Does this follow "one parameter = one feature"?

## Development setup

```bash
# Clone the repository
git clone https://github.com/jmbarrancoml/pureagents.git
cd pureagents

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install with dev dependencies
pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

Tests should work without API keys. Use mocking for external services.

## Code style

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
ruff check .
ruff format .
```

Guidelines:
- Type hints everywhere
- Docstrings for public APIs only
- 88 character line limit
- British English in documentation

## Submitting changes

1. Fork the repository
2. Create a branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests and linting
5. Commit with a clear message
6. Push and open a pull request

## Pull request checklist

- [ ] Tests pass
- [ ] Linting passes
- [ ] Documentation updated (if applicable)
- [ ] Changelog entry added (if applicable)

## Reporting issues

When reporting bugs, please include:

- Python version
- pureagents version
- Provider being used (Mistral/OpenAI/Anthropic)
- Minimal code to reproduce
- Full error traceback

## Questions?

Open an issue or start a discussion.
