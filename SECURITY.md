# Security policy

## Supported versions

| Version | Supported          |
| ------- | ------------------ |
| 0.x.x   | :white_check_mark: |

## Reporting a vulnerability

If you discover a security vulnerability, please report it privately:

1. **Do not** open a public issue
2. Email jose@ederspark.com with details
3. Include steps to reproduce if possible
4. Allow up to 48 hours for an initial response

We will:

- Acknowledge receipt within 48 hours
- Provide an estimated timeline for a fix
- Notify you when the issue is resolved
- Credit you in the release notes (unless you prefer anonymity)

## Security considerations

When using pureagents:

- **API keys**: Never commit API keys. Use environment variables.
- **Tool execution**: Tools execute arbitrary code. Only use trusted tools.
- **User input**: Sanitise user input before passing to agents.
- **Structured outputs**: Validate structured outputs before using in security-sensitive contexts.

## Scope

This policy covers the pureagents Python package. Third-party dependencies and LLM provider APIs have their own security policies.
