"""Cross-encoder reranker adapter (local, CPU, via fastembed).

BI-ENCODER vs CROSS-ENCODER — the trade-off this module exists to exploit:
- Retrieval (bi-encoder): query and documents are embedded SEPARATELY, so
  document vectors are precomputed once and search is a fast nearest-neighbor
  lookup over the whole corpus. Cheap, scalable — but the model never sees
  query and document TOGETHER, so scoring is approximate.
- Reranking (cross-encoder): the model reads query+document as ONE input and
  scores their actual interaction. Far more precise — and far too slow to run
  against a whole corpus, since nothing can be precomputed.

The production pattern is therefore a funnel: retrieve a generous candidate
set with the cheap method, then let the expensive method pick the best few.
"""

from typing import Sequence

from fastembed.rerank.cross_encoder import TextCrossEncoder

from nimbusdesk.domain.knowledge import RetrievedChunk


class FastEmbedReranker:
    def __init__(self, model_name: str) -> None:
        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(
        self, query: str, results: Sequence[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not results:
            return []
        # Give the cross-encoder the same context-enriched text we embed:
        # a bare chunk ("it expires after 14 days") is as ambiguous to a
        # reranker as it is to an embedding model.
        documents = [
            f"{r.chunk.title} — {r.chunk.section}\n{r.chunk.text}" for r in results
        ]
        scores = list(self._model.rerank(query, documents))
        ranked = sorted(range(len(results)), key=lambda i: scores[i], reverse=True)
        return [
            RetrievedChunk(chunk=results[i].chunk, score=float(scores[i]))
            for i in ranked[:top_n]
        ]
