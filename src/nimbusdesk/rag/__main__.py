"""Developer entry point for the RAG pipeline.

    uv run python -m nimbusdesk.rag ingest            # index data/seed into Qdrant
    uv run python -m nimbusdesk.rag search "query"    # debug retrieval (no LLM)
    uv run python -m nimbusdesk.rag ask "question"    # full grounded pipeline (LLM)

This is a dev utility, not the product interface (that's phase 9's FastAPI +
CLI). It's also the COMPOSITION ROOT: the one place where ports meet adapters —
settings are read, FastEmbed/Qdrant/Anthropic are constructed and injected into
the vendor-agnostic pipeline classes. `search` needs no API key; `ask` does.
"""

import argparse
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from nimbusdesk.infrastructure.embeddings import FastEmbedEmbedder, FastEmbedSparseEmbedder
from nimbusdesk.infrastructure.settings import Settings, get_settings
from nimbusdesk.infrastructure.vector_store import QdrantVectorIndex
from nimbusdesk.rag.ingestion import IngestionPipeline
from nimbusdesk.rag.retrieval import Retriever

# Windows consoles often default to legacy cp1252, which can't print characters
# like "≥" that appear in our documents. Force UTF-8 (replacing anything truly
# unprintable) so a snippet's content never crashes a dev tool.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

DEFAULT_DOCS_DIR = Path("data/seed")


def _build_retrieval(
    settings: Settings,
) -> tuple[FastEmbedEmbedder, FastEmbedSparseEmbedder, QdrantVectorIndex]:
    dense = FastEmbedEmbedder(
        model_name=settings.embedding_model_name,
        dimension=settings.embedding_dimension,
    )
    sparse = FastEmbedSparseEmbedder(model_name=settings.sparse_model_name)
    index = QdrantVectorIndex(
        client=QdrantClient(url=settings.qdrant_url),
        collection=settings.qdrant_collection,
        dimension=settings.embedding_dimension,
    )
    return dense, sparse, index


def _cmd_ingest(docs_dir: Path) -> None:
    dense, sparse, index = _build_retrieval(get_settings())
    report = IngestionPipeline(dense, sparse, index).run(docs_dir)
    print(f"Ingested {report.documents} documents as {report.chunks} chunks.")


def _cmd_search(query: str, k: int) -> None:
    dense, sparse, index = _build_retrieval(get_settings())
    results = Retriever(dense, sparse, index).search(query, k=k)
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        snippet = chunk.text[:180].replace("\n", " ")
        print(f"\n#{rank}  score={result.score:.3f}  [{chunk.doc_id} — {chunk.section}]")
        print(f"    {snippet}...")


def _cmd_ask(question: str) -> None:
    # Imports here so `search`/`ingest` never pay for (or require) LLM setup.
    from nimbusdesk.infrastructure.anthropic_llm import AnthropicProvider
    from nimbusdesk.infrastructure.reranker import FastEmbedReranker
    from nimbusdesk.llm.tracking import UsageTracker
    from nimbusdesk.rag.answering import AnswerGenerator
    from nimbusdesk.rag.pipeline import GroundedRagPipeline
    from nimbusdesk.rag.rewrite import QueryRewriter
    from nimbusdesk.rag.self_check import FaithfulnessChecker

    settings = get_settings()
    api_key = (
        settings.anthropic_api_key.get_secret_value()
        if settings.anthropic_api_key
        else None
    )
    # Model-tier routing (see llm/__init__.py): cheap model for the mechanical
    # steps, strong model for the customer-facing answer. Each is wrapped in a
    # UsageTracker so the pipeline can report the true token cost per answer.
    fast_llm = UsageTracker(AnthropicProvider(api_key=api_key, model=settings.nimbus_model_fast))
    strong_llm = UsageTracker(
        AnthropicProvider(api_key=api_key, model=settings.nimbus_model_strong)
    )

    dense, sparse, index = _build_retrieval(settings)
    pipeline = GroundedRagPipeline(
        rewriter=QueryRewriter(fast_llm),
        retriever=Retriever(dense, sparse, index),
        reranker=FastEmbedReranker(settings.reranker_model_name),
        generator=AnswerGenerator(strong_llm),
        checker=FaithfulnessChecker(fast_llm),
        usage_trackers=[fast_llm, strong_llm],
        candidates=settings.retrieval_candidates,
        top_k=settings.retrieval_top_k,
    )

    result = pipeline.ask(question)

    print(f"\n{result.answer}\n")
    if result.citations:
        print("Sources:")
        for citation in result.citations:
            print(f"  [{citation.marker}] {citation.title} — {citation.section}")
    from nimbusdesk.observability.cost import estimate_usd

    status = "grounded" if result.grounded else "NOT grounded — needs human review"
    # Rough split: attribute all tokens to the strong model — an upper-bound
    # estimate (fast-tier calls are cheaper), which is the safe direction.
    cost = estimate_usd(settings.nimbus_model_strong, result.input_tokens, result.output_tokens)
    print(
        f"\n({status} | tokens: {result.input_tokens} in / {result.output_tokens} out "
        f"| est. cost: ${cost:.4f})"
    )
    if result.notes:
        print(f"note: {result.notes}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="nimbusdesk.rag")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Index a directory of .md files into Qdrant")
    ingest.add_argument("--dir", type=Path, default=DEFAULT_DOCS_DIR)

    search = sub.add_parser("search", help="Debug retrieval: hybrid search, no LLM")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5)

    ask = sub.add_parser("ask", help="Full grounded answer with citations (needs API key)")
    ask.add_argument("question")

    args = parser.parse_args()
    if args.command == "ingest":
        _cmd_ingest(args.dir)
    elif args.command == "search":
        _cmd_search(args.query, args.k)
    else:
        # Missing/invalid API key is an expected user state, not a bug: exit
        # with remediation steps instead of a stack trace. (This is the
        # composition root — the one place outside infrastructure/ allowed to
        # know the vendor's exception types.)
        from anthropic import AuthenticationError

        from nimbusdesk.infrastructure.anthropic_llm import MissingApiKeyError

        try:
            _cmd_ask(args.question)
        except MissingApiKeyError as error:
            print(f"error: {error}", file=sys.stderr)
            raise SystemExit(1) from None
        except AuthenticationError:
            print(
                "error: Anthropic rejected the API key (401). Check the "
                "ANTHROPIC_API_KEY value in your .env — it may be a placeholder, "
                "revoked, or truncated.",
                file=sys.stderr,
            )
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
