"""Golden-dataset evaluation suite.

    uv run python evals/run_eval.py                  # everything possible
    uv run python evals/run_eval.py --suite retrieval

WHY OFFLINE EVALS EXIST (vs the runtime self-check)
The runtime faithfulness check catches problems per-answer, in production.
This suite catches REGRESSIONS per-change, in CI: swap the embedding model,
tweak the chunker, reword the triage prompt — and the scores tell you what
got better or worse BEFORE customers do. A golden dataset is version-controlled
product knowledge: "these questions must keep working."

THREE SUITES, THREE COSTS
- retrieval    (free, always runs): hit@3 over the hybrid+rerank funnel,
  using local embeddings and an in-process Qdrant. The eval every PR can run.
- routing      (needs API key, ~cents): does triage+policy send each golden
  ticket to the right specialist?
- faithfulness (needs API key, ~cents): full grounded pipeline over golden
  questions; scored by answers being grounded AND cited (LLM-as-judge via our
  own FaithfulnessChecker).

Exit code is non-zero when any executed suite scores below its threshold —
that's what makes this CI-able (ADR-11, revised: hand-rolled harness reusing
our own components instead of DeepEval; see ARCHITECTURE.md).
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).parents[1]
DATASET_DIR = REPO / "evals" / "golden_dataset"
REPORT_PATH = REPO / "evals" / "report.json"
SEED_DIR = REPO / "data" / "seed"

THRESHOLDS = {"retrieval": 0.85, "routing": 0.80, "faithfulness": 0.75}

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(name: str) -> list[dict]:
    path = DATASET_DIR / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _build_retrieval_stack():
    """Self-contained: local embeddings + in-process Qdrant, no docker needed."""
    from qdrant_client import QdrantClient

    from nimbusdesk.infrastructure.embeddings import (
        FastEmbedEmbedder,
        FastEmbedSparseEmbedder,
    )
    from nimbusdesk.infrastructure.reranker import FastEmbedReranker
    from nimbusdesk.infrastructure.settings import get_settings
    from nimbusdesk.infrastructure.vector_store import QdrantVectorIndex
    from nimbusdesk.rag.ingestion import IngestionPipeline
    from nimbusdesk.rag.retrieval import Retriever

    settings = get_settings()
    dense = FastEmbedEmbedder(settings.embedding_model_name, settings.embedding_dimension)
    sparse = FastEmbedSparseEmbedder(settings.sparse_model_name)
    index = QdrantVectorIndex(QdrantClient(":memory:"), "eval_kb", settings.embedding_dimension)
    IngestionPipeline(dense, sparse, index).run(SEED_DIR)
    return Retriever(dense, sparse, index), FastEmbedReranker(settings.reranker_model_name)


def eval_retrieval() -> dict:
    """Metric: hit@3 after the full funnel (hybrid retrieve 20 -> rerank 3).
    Retrieval quality bounds everything downstream, so it gets the strictest
    threshold."""
    retriever, reranker = _build_retrieval_stack()
    cases = _load("retrieval.jsonl")
    failures = []
    for case in cases:
        candidates = retriever.search(case["question"], k=20)
        top = reranker.rerank(case["question"], candidates, top_n=3)
        found = [r.chunk.doc_id for r in top]
        if case["expected_doc"] not in found:
            failures.append({**case, "got": found})
    return _suite_result("retrieval", len(cases), failures)


def _build_fast_llm():
    from nimbusdesk.infrastructure.anthropic_llm import AnthropicProvider
    from nimbusdesk.infrastructure.settings import get_settings

    settings = get_settings()
    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
    return AnthropicProvider(api_key=key, model=settings.nimbus_model_fast)


def eval_routing() -> dict:
    """Metric: accuracy of triage (LLM) + supervisor policy (code) end to end.
    'escalation' expectations cover urgent/ambiguous tickets that must reach
    humans — misrouting those is the expensive failure."""
    from nimbusdesk.agents.state import SupportState
    from nimbusdesk.agents.supervisor import route_from_supervisor
    from nimbusdesk.agents.triage import TriageAgent

    triage_agent = TriageAgent(_build_fast_llm())
    cases = _load("routing.jsonl")
    failures = []
    for case in cases:
        decision = triage_agent.triage(case["question"])
        state = SupportState(question=case["question"], triage=decision, supervisor_visits=1)
        route = route_from_supervisor(state)
        if route != case["expected_route"]:
            failures.append(
                {**case, "got": route, "triage": decision.model_dump(mode="json")}
            )
    return _suite_result("routing", len(cases), failures)


def eval_faithfulness() -> dict:
    """Metric: fraction of golden questions whose pipeline answer is BOTH
    grounded (passes the independent self-check) AND carries citations."""
    from nimbusdesk.infrastructure.anthropic_llm import AnthropicProvider
    from nimbusdesk.infrastructure.settings import get_settings
    from nimbusdesk.rag.answering import AnswerGenerator
    from nimbusdesk.rag.pipeline import GroundedRagPipeline
    from nimbusdesk.rag.rewrite import QueryRewriter
    from nimbusdesk.rag.self_check import FaithfulnessChecker

    settings = get_settings()
    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
    fast = AnthropicProvider(api_key=key, model=settings.nimbus_model_fast)
    strong = AnthropicProvider(api_key=key, model=settings.nimbus_model_strong)
    retriever, reranker = _build_retrieval_stack()
    pipeline = GroundedRagPipeline(
        rewriter=QueryRewriter(fast),
        retriever=retriever,
        reranker=reranker,
        generator=AnswerGenerator(strong),
        checker=FaithfulnessChecker(fast),
    )

    cases = _load("faithfulness.jsonl")
    failures = []
    for case in cases:
        answer = pipeline.ask(case["question"])
        if not (answer.grounded and answer.citations):
            failures.append(
                {**case, "grounded": answer.grounded, "citations": len(answer.citations),
                 "notes": answer.notes}
            )
    return _suite_result("faithfulness", len(cases), failures)


def _suite_result(name: str, total: int, failures: list[dict]) -> dict:
    score = (total - len(failures)) / total if total else 0.0
    return {
        "suite": name,
        "cases": total,
        "passed": total - len(failures),
        "score": round(score, 3),
        "threshold": THRESHOLDS[name],
        "ok": score >= THRESHOLDS[name],
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", choices=["retrieval", "routing", "faithfulness", "all"], default="all"
    )
    args = parser.parse_args()

    from nimbusdesk.infrastructure.settings import get_settings

    has_key = get_settings().anthropic_api_key is not None
    suites = []

    if args.suite in ("retrieval", "all"):
        suites.append(eval_retrieval())
    for name, fn in (("routing", eval_routing), ("faithfulness", eval_faithfulness)):
        if args.suite in (name, "all"):
            if has_key:
                suites.append(fn())
            else:
                print(f"[skip] {name}: needs ANTHROPIC_API_KEY in .env")

    print("\n=== eval report " + "=" * 44)
    for result in suites:
        status = "PASS" if result["ok"] else "FAIL"
        print(
            f"{status}  {result['suite']:<13} {result['passed']}/{result['cases']} "
            f"(score {result['score']:.0%}, threshold {result['threshold']:.0%})"
        )
        for failure in result["failures"]:
            print(f"      miss: {json.dumps(failure, ensure_ascii=False)[:160]}")
    print("=" * 60)

    REPORT_PATH.write_text(json.dumps(suites, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report written to {REPORT_PATH.relative_to(REPO)}")

    if any(not result["ok"] for result in suites):
        sys.exit(1)


if __name__ == "__main__":
    main()
