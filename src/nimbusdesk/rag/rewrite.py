"""Query rewriting — the first "agentic" control point of the pipeline.

WHY: users write for humans ("hey, my boss is asking if we can force everyone
to use 2FA?"), but retrieval works best on dense, keyword-rich search phrasing
("enforce two-factor authentication workspace-wide"). A cheap LLM call closes
that gap before retrieval runs.

DESIGN DECISIONS WORTH DEFENDING
- FAST/CHEAP MODEL TIER: rewriting is mechanical; burning the strong model
  here roughly doubles pipeline cost for no measurable gain.
- GRACEFUL DEGRADATION: if the LLM call fails (rate limit, network), we fall
  back to the raw question. A slightly worse search beats a crashed pipeline —
  the LLM is an OPTIMIZATION here, not a dependency. Failing open is correct
  for quality steps; phase 7 shows the opposite (failing CLOSED) for safety
  steps. Knowing which is which is the actual skill.
"""

import logging

from nimbusdesk.llm.ports import LLMProvider, Message

logger = logging.getLogger(__name__)

_SYSTEM = """You turn customer-support questions into effective search queries.

Rewrite the user's message as one short query for searching a technical
knowledge base: keep product terms, error codes and identifiers EXACTLY as
written, drop greetings and filler, expand vague phrasing into concrete
technical terms. Reply with the query only — no quotes, no explanation."""


class QueryRewriter:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def rewrite(self, question: str) -> str:
        try:
            completion = self._llm.complete(
                system=_SYSTEM,
                messages=[Message(role="user", content=question)],
                max_tokens=100,
            )
        except Exception as error:
            # One-line warning, no traceback: this fallback is EXPECTED under
            # LLM outages, and a screaming stack trace would train operators
            # to ignore logs. The generation step will surface real failures.
            logger.warning(
                "query rewrite failed (%s: %s); falling back to raw question",
                type(error).__name__,
                error,
            )
            return question

        rewritten = completion.text.strip().strip('"')
        # An empty or bloated rewrite is worse than none — sanity-bound it.
        if not rewritten or len(rewritten) > 4 * max(len(question), 80):
            return question
        return rewritten
