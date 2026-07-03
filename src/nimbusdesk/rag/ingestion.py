"""Ingestion pipeline: load -> chunk -> embed -> index.

This module only ORCHESTRATES — every real capability arrives via a port
(Embedder, VectorIndex), so the pipeline itself is testable with fakes in
microseconds and never knows which vendor is behind it.
"""

from pathlib import Path

from pydantic import BaseModel

from nimbusdesk.domain.knowledge import DocumentChunk
from nimbusdesk.rag.chunking import chunk_document
from nimbusdesk.rag.loading import load_markdown_dir
from nimbusdesk.rag.ports import Embedder, VectorIndex


class IngestionReport(BaseModel):
    documents: int
    chunks: int


def embedding_input(chunk: DocumentChunk) -> str:
    """The text we actually embed: heading context + body.

    A chunk that says "it expires after 14 days" embeds poorly on its own.
    Prefixing "Billing, plans and refunds — Refund policy" restores the topic
    signal the chunk lost when it was cut from its document. This is a light
    version of "contextual retrieval" (the heavyweight variant uses an LLM to
    write a bespoke context sentence per chunk — better, but costs one LLM
    call per chunk; overkill at our corpus size).

    NOTE: we embed the enriched text but STORE the raw text — what the user
    sees as a citation should be exactly what the document says.
    """
    return f"{chunk.title} — {chunk.section}\n\n{chunk.text}"


class IngestionPipeline:
    def __init__(self, embedder: Embedder, index: VectorIndex) -> None:
        self._embedder = embedder
        self._index = index

    def run(self, docs_dir: Path) -> IngestionReport:
        documents = load_markdown_dir(docs_dir)
        chunks = [chunk for doc in documents for chunk in chunk_document(doc)]

        # One batched call: embedding models amortize startup cost over the
        # batch; embedding chunk-by-chunk is the classic 10x slowdown.
        vectors = self._embedder.embed_passages([embedding_input(c) for c in chunks])

        self._index.ensure_ready()
        self._index.upsert(chunks, vectors)
        return IngestionReport(documents=len(documents), chunks=len(chunks))
