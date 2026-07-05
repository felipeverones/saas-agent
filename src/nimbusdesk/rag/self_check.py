"""Faithfulness self-check — does the answer actually follow from the sources?

THE PROBLEM THIS SOLVES
Generation can go wrong even with perfect retrieval: the model can blend its
training knowledge into the answer ("refunds usually take 30 days" — usually
WHERE?). This step re-reads the draft against the retrieved documents and
flags every claim without support. It's LLM-as-judge, pointed at ourselves.

WHY A SEPARATE, CHEAP CALL (not "be careful" in the generation prompt)
Asking the generator to police itself while writing doesn't work reliably —
verification is a different task from generation, and a fresh pass with a
narrow rubric ("is THIS sentence supported by THESE documents?") is far more
accurate than one prompt juggling both jobs. The checker uses the fast model:
verification is closer to reading comprehension than to reasoning.

FAILURE POLICY (worth quoting in interviews)
- Unsupported claims found -> pipeline retries ONCE with a revision note, then
  gives up and marks the answer grounded=False (callers must warn/escalate).
  Bounded retries: quality loops without a budget become infinite loops.
- Checker itself fails/unparseable -> we FAIL OPEN (answer ships, flagged in
  notes). For an informational answer, availability beats perfection; for
  ACTIONS (refunds, phase 7) the same failure fails CLOSED. Same machinery,
  opposite defaults — the asymmetry is the design.
"""

import logging

from pydantic import BaseModel

from nimbusdesk.domain.knowledge import RetrievedChunk
from nimbusdesk.llm.json_parsing import extract_json_object
from nimbusdesk.llm.ports import LLMProvider, Message
from nimbusdesk.rag.answering import build_context

logger = logging.getLogger(__name__)

_SYSTEM = """You are a fact-checking auditor. You receive numbered <document> \
blocks and a draft answer written from them.

List every FACTUAL claim in the draft that is NOT directly supported by the \
documents. Paraphrases are fine; invented numbers, policies, or details are \
not. Ignore hedges, greetings and "contact support" suggestions.

Reply with ONLY this JSON, no other text:
{"unsupported_claims": ["<claim 1>", "<claim 2>"]}
Return {"unsupported_claims": []} if everything is supported."""


class SelfCheckResult(BaseModel):
    grounded: bool
    unsupported_claims: list[str] = []
    inconclusive: bool = False


class FaithfulnessChecker:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def check(self, answer_text: str, chunks: list[RetrievedChunk]) -> SelfCheckResult:
        prompt = f"{build_context(chunks)}\n\nDraft answer to audit:\n{answer_text}"
        try:
            completion = self._llm.complete(
                system=_SYSTEM,
                messages=[Message(role="user", content=prompt)],
                max_tokens=500,
            )
            claims = self._parse(completion.text)
        except Exception as error:
            logger.warning(
                "faithfulness check errored (%s: %s); failing open",
                type(error).__name__,
                error,
            )
            return SelfCheckResult(grounded=True, inconclusive=True)

        if claims is None:
            return SelfCheckResult(grounded=True, inconclusive=True)
        return SelfCheckResult(grounded=not claims, unsupported_claims=claims)

    @staticmethod
    def _parse(text: str) -> list[str] | None:
        """None = unparseable (inconclusive), never an exception."""
        data = extract_json_object(text)
        if data is None:
            return None
        claims = data.get("unsupported_claims")
        if isinstance(claims, list):
            return [str(c) for c in claims]
        return None
