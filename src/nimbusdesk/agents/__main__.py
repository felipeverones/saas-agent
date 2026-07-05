"""Developer entry point for the single support agent (phase 3).

    uv run python -m nimbusdesk.agents "my sync is slow, is something down?"

Prints the final answer AND the step trace (which tools ran, with what
arguments, what they observed) — watching the reason->act->observe loop is
the whole point of this phase.
"""

import argparse
import sys

from qdrant_client import QdrantClient

from nimbusdesk.agents.support_agent import build_support_agent
from nimbusdesk.infrastructure.anthropic_llm import AnthropicProvider, MissingApiKeyError
from nimbusdesk.infrastructure.embeddings import FastEmbedEmbedder, FastEmbedSparseEmbedder
from nimbusdesk.infrastructure.reranker import FastEmbedReranker
from nimbusdesk.infrastructure.settings import get_settings
from nimbusdesk.infrastructure.vector_store import QdrantVectorIndex
from nimbusdesk.llm.tracking import UsageTracker
from nimbusdesk.rag.retrieval import Retriever

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def main() -> None:
    parser = argparse.ArgumentParser(prog="nimbusdesk.agents")
    parser.add_argument("question")
    args = parser.parse_args()

    settings = get_settings()
    api_key = (
        settings.anthropic_api_key.get_secret_value()
        if settings.anthropic_api_key
        else None
    )

    try:
        llm = UsageTracker(
            AnthropicProvider(api_key=api_key, model=settings.nimbus_model_strong)
        )
    except MissingApiKeyError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    retriever = Retriever(
        FastEmbedEmbedder(settings.embedding_model_name, settings.embedding_dimension),
        FastEmbedSparseEmbedder(settings.sparse_model_name),
        QdrantVectorIndex(
            client=QdrantClient(url=settings.qdrant_url),
            collection=settings.qdrant_collection,
            dimension=settings.embedding_dimension,
        ),
    )
    agent = build_support_agent(
        llm, retriever, FastEmbedReranker(settings.reranker_model_name)
    )

    from anthropic import AuthenticationError

    try:
        result = agent.run(args.question)
    except AuthenticationError:
        print(
            "error: Anthropic rejected the API key (401). Check ANTHROPIC_API_KEY "
            "in your .env — it may be a placeholder, revoked, or truncated.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    if result.steps:
        print("--- agent steps " + "-" * 44)
        for i, step in enumerate(result.steps, start=1):
            flag = " [error]" if step.is_error else ""
            print(f"{i}. {step.tool}({step.arguments}){flag}")
            preview = step.observation[:160].replace("\n", " ")
            print(f"   -> {preview}")
        print("-" * 60)

    print(f"\n{result.answer}\n")
    limit_note = " | HIT ITERATION LIMIT" if result.hit_iteration_limit else ""
    print(
        f"({result.iterations} iteration(s), {len(result.steps)} tool call(s), "
        f"tokens: {llm.input_tokens} in / {llm.output_tokens} out{limit_note})"
    )


if __name__ == "__main__":
    main()
