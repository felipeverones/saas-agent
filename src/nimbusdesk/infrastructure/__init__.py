"""Infrastructure layer — adapters to the outside world (the "plugins" of the app).

WHAT LIVES HERE
Concrete implementations of external concerns: the Anthropic LLM client, the
Qdrant client, SQLite persistence, settings loading. Everything with an API key,
a network socket or a disk path belongs here.

THE PORTS & ADAPTERS IDEA
Inner layers define the interface they NEED (a "port", e.g. `LLMProvider` with a
`complete()` method); this layer provides what EXISTS (an "adapter", e.g.
`AnthropicProvider`). The dependency points inward: agents know the port, never
the vendor. Two payoffs:
1. Tests: a `FakeLLMProvider` returning canned responses makes the entire test
   suite free, fast and deterministic — no API keys in CI, ever.
2. Portability: "how would you migrate providers?" -> "one new adapter, zero
   changes to agents." That answer is this package.

`settings.py` (pydantic-settings) validates ALL env config at import time:
misconfiguration should kill the process at startup, not a customer conversation
at 3am.
"""
