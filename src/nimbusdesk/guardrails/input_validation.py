"""Input validation — the first gate anything from the outside passes.

TWO-TIER POLICY (the design decision worth remembering)
- STRUCTURAL problems (empty, absurdly long, control characters) -> REJECT.
  These are never legitimate support messages; rejecting costs nothing.
- SUSPICIOUS CONTENT (injection-looking phrases) -> FLAG, DON'T BLOCK.
  Heuristics have false positives, and a customer legitimately quoting an
  error ("the bot told me to ignore previous instructions??") must not be
  locked out. Flags travel in the state for triage/escalation/audit to weigh.
Blocking on weak signals punishes users; flagging keeps humans informed.
"""

import re
import unicodedata

from pydantic import BaseModel, Field

MAX_MESSAGE_CHARS = 4000

# Classic direct-injection phrasings. Deliberately a COARSE net: this list
# catches script kiddies and accidents, not determined attackers — the real
# defenses are structural (schema-validated tools, consent gates, HITL).
# Never rely on phrase-matching as your only line.
_INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore (all |any |your )?(previous|prior|above) (instructions|rules|prompts)",
        r"disregard (all |your )?(previous|prior|system) (instructions|rules|prompts)",
        r"you are now [a-z]",
        r"system prompt",
        r"reveal (your|the) (instructions|prompt|rules)",
        r"\bDAN mode\b",
        r"pretend (you are|to be) (not )?an? (ai|assistant|llm)",
    )
]


class InputCheck(BaseModel):
    ok: bool
    sanitized: str = ""
    rejection_reason: str | None = None
    flags: list[str] = Field(default_factory=list)


def validate_customer_message(raw: str) -> InputCheck:
    # Strip control characters (except newlines/tabs): they carry no meaning
    # in a support message and are a classic smuggling channel.
    sanitized = "".join(
        ch for ch in raw if ch in "\n\t" or unicodedata.category(ch)[0] != "C"
    ).strip()

    if not sanitized:
        return InputCheck(ok=False, rejection_reason="empty message")
    if len(sanitized) > MAX_MESSAGE_CHARS:
        return InputCheck(
            ok=False,
            rejection_reason=f"message exceeds {MAX_MESSAGE_CHARS} characters",
        )

    flags = [
        f"injection-pattern: {pattern.pattern[:40]}"
        for pattern in _INJECTION_PATTERNS
        if pattern.search(sanitized)
    ]
    return InputCheck(ok=True, sanitized=sanitized, flags=flags)
