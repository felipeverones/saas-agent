"""The grounded RAG pipeline — where all phase-2 pieces click together.

    question -> rewrite -> hybrid retrieve -> rerank -> generate -> self-check
                                ^                                      |
                                +---- one corrective round at most ----+

WHAT MAKES THIS "AGENTIC" RAG rather than a fixed pipe: the system inspects
its own intermediate results and CHANGES ITS BEHAVIOR — rewriting the query
before searching, and re-retrieving/regenerating when the self-check finds
unsupported claims. Naive RAG executes; agentic RAG executes AND verifies.

The corrective loop is BOUNDED (one round). Unbounded self-correction is how
agent systems melt into infinite loops; every loop in this codebase has an
explicit budget — the same principle returns as max_iterations in phase 4.
"""

from typing import Sequence

from nimbusdesk.domain.knowledge import GroundedAnswer, RetrievedChunk
from nimbusdesk.llm.tracking import UsageTracker
from nimbusdesk.observability.tracing import span
from nimbusdesk.rag.answering import AnswerGenerator
from nimbusdesk.rag.ports import Reranker
from nimbusdesk.rag.retrieval import Retriever
from nimbusdesk.rag.rewrite import QueryRewriter
from nimbusdesk.rag.self_check import FaithfulnessChecker

MAX_CORRECTION_ROUNDS = 1


class GroundedRagPipeline:
    def __init__(
        self,
        rewriter: QueryRewriter,
        retriever: Retriever,
        reranker: Reranker,
        generator: AnswerGenerator,
        checker: FaithfulnessChecker,
        usage_trackers: Sequence[UsageTracker] = (),
        candidates: int = 20,
        top_k: int = 5,
    ) -> None:
        self._rewriter = rewriter
        self._retriever = retriever
        self._reranker = reranker
        self._generator = generator
        self._checker = checker
        # The trackers wrap the SAME provider instances the components above
        # hold — so reading them after a run gives the true total across every
        # LLM call this answer cost, with zero bookkeeping in the components.
        self._usage_trackers = list(usage_trackers)
        self._candidates = candidates
        self._top_k = top_k

    def ask(self, question: str) -> GroundedAnswer:
        with span("rag.ask", question=question[:200]) as ask_span:
            answer = self._ask(question)
            ask_span.set_attribute("rag.grounded", answer.grounded)
            ask_span.set_attribute("rag.citations", len(answer.citations))
            return answer

    def _ask(self, question: str) -> GroundedAnswer:
        tokens_before = self._usage_snapshot()

        with span("rag.rewrite"):
            search_query = self._rewriter.rewrite(question)
        chunks = self._retrieve(search_query)
        with span("rag.generate"):
            draft = self._generator.generate(question, chunks)

        with span("rag.self_check") as check_span:
            check = self._checker.check(draft.answer, chunks)
            check_span.set_attribute("rag.check_grounded", check.grounded)
        rounds = 0
        while not check.grounded and rounds < MAX_CORRECTION_ROUNDS:
            rounds += 1
            # Second chance, smarter: search again INCLUDING the unsupported
            # claims as query terms — if evidence exists, this surfaces it; the
            # revision note tells the generator exactly what to fix or drop.
            claims_text = "; ".join(check.unsupported_claims)
            chunks = self._retrieve(f"{search_query} {claims_text}")
            draft = self._generator.generate(question, chunks, revision_note=claims_text)
            check = self._checker.check(draft.answer, chunks)

        notes = None
        if check.inconclusive:
            notes = "Self-check was inconclusive; answer shipped unverified."
        elif not check.grounded:
            notes = "Unsupported after retry: " + "; ".join(check.unsupported_claims)

        tokens_after = self._usage_snapshot()
        return draft.model_copy(
            update={
                "grounded": check.grounded,
                "notes": notes,
                "input_tokens": tokens_after[0] - tokens_before[0],
                "output_tokens": tokens_after[1] - tokens_before[1],
            }
        )

    def _usage_snapshot(self) -> tuple[int, int]:
        return (
            sum(t.input_tokens for t in self._usage_trackers),
            sum(t.output_tokens for t in self._usage_trackers),
        )

    def _retrieve(self, query: str) -> list[RetrievedChunk]:
        # The funnel: wide net first (recall), precise re-scoring after
        # (precision). Retrieval mistakes here are unrecoverable downstream —
        # the generator can only cite what this step surfaces.
        candidates = self._retriever.search(query, k=self._candidates)
        with span("rag.rerank", candidates=len(candidates), top_n=self._top_k):
            return self._reranker.rerank(query, candidates, top_n=self._top_k)
