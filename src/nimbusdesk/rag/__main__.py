"""Developer entry point for the RAG pipeline.

    uv run python -m nimbusdesk.rag ingest            # index data/seed into Qdrant
    uv run python -m nimbusdesk.rag search "query"    # eyeball retrieval quality

This is a dev utility, not the product interface (that's phase 9's FastAPI +
CLI). It's also the COMPOSITION ROOT for the pipeline: the one place where
ports meet adapters — settings are read, FastEmbed and Qdrant are constructed,
and injected into the vendor-agnostic pipeline classes.
"""

import argparse
from pathlib import Path

from qdrant_client import QdrantClient

from nimbusdesk.infrastructure.embeddings import FastEmbedEmbedder
from nimbusdesk.infrastructure.settings import get_settings
from nimbusdesk.infrastructure.vector_store import QdrantVectorIndex
from nimbusdesk.rag.ingestion import IngestionPipeline
from nimbusdesk.rag.retrieval import Retriever

DEFAULT_DOCS_DIR = Path("data/seed")


def _build_components() -> tuple[FastEmbedEmbedder, QdrantVectorIndex]:
    settings = get_settings()
    embedder = FastEmbedEmbedder(
        model_name=settings.embedding_model_name,
        dimension=settings.embedding_dimension,
    )
    index = QdrantVectorIndex(
        client=QdrantClient(url=settings.qdrant_url),
        collection=settings.qdrant_collection,
        dimension=settings.embedding_dimension,
    )
    return embedder, index


def _cmd_ingest(docs_dir: Path) -> None:
    embedder, index = _build_components()
    report = IngestionPipeline(embedder, index).run(docs_dir)
    print(f"Ingested {report.documents} documents as {report.chunks} chunks.")


def _cmd_search(query: str, k: int) -> None:
    embedder, index = _build_components()
    results = Retriever(embedder, index).search(query, k=k)
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        snippet = chunk.text[:180].replace("\n", " ")
        print(f"\n#{rank}  score={result.score:.3f}  [{chunk.doc_id} — {chunk.section}]")
        print(f"    {snippet}...")


def main() -> None:
    parser = argparse.ArgumentParser(prog="nimbusdesk.rag")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Index a directory of .md files into Qdrant")
    ingest.add_argument("--dir", type=Path, default=DEFAULT_DOCS_DIR)

    search = sub.add_parser("search", help="Run a query against the index")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5)

    args = parser.parse_args()
    if args.command == "ingest":
        _cmd_ingest(args.dir)
    else:
        _cmd_search(args.query, args.k)


if __name__ == "__main__":
    main()
