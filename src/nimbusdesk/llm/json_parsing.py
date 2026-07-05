"""Tolerant JSON extraction from LLM replies.

LLMs asked for "only JSON" still love to add prose ("Here is the JSON: ...")
or wrap output in markdown fences. Every JSON-consuming step needs the same
tolerance, so it lives here once. Returns None instead of raising: the CALLER
decides what a parse failure means (fail open, fallback value, escalate) —
that policy is context-dependent and must not be buried in a parsing utility.
"""

import json
import re

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> dict | None:
    """Best-effort: find the outermost {...} block and parse it."""
    match = _JSON_BLOCK.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
