"""Domain models for the knowledge base.

WHY THESE LIVE IN THE DOMAIN LAYER
"A piece of documentation, traceable to its source" is a business concept —
support answers must cite where they came from regardless of which vector
database or embedding model we use. HOW chunks are embedded/stored is an
infrastructure concern and is deliberately absent here: note there is no
`vector` field on DocumentChunk. Vectors are a storage detail that would drag
numpy/qdrant types into the domain.
"""

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """A whole knowledge-base article, as loaded from disk."""

    doc_id: str = Field(description="Stable slug, e.g. the filename stem")
    title: str
    path: str
    text: str


class DocumentChunk(BaseModel):
    """A retrieval-sized slice of a document.

    `section` keeps the heading trail (e.g. "Refund policy") so a chunk still
    carries its context after being separated from the full article — both for
    better embeddings and for human-readable citations.
    """

    chunk_id: str = Field(description="Deterministic UUID — same input, same id")
    doc_id: str
    title: str
    section: str
    text: str
    position: int = Field(ge=0, description="Order of the chunk within its document")


class RetrievedChunk(BaseModel):
    """A chunk returned by a search, with its relevance score.

    This is the unit that citations are built from: (title, section, text)
    gives full provenance for any claim in a generated answer.
    """

    chunk: DocumentChunk
    score: float


class Citation(BaseModel):
    """Provenance for one source used in a generated answer.

    `marker` matches the inline [n] reference in the answer text, so a UI can
    render clickable footnotes. An answer without citations is, by policy,
    not trustworthy (see GroundedAnswer.grounded).
    """

    marker: int = Field(ge=1)
    doc_id: str
    title: str
    section: str
    snippet: str = Field(description="Verbatim excerpt from the source chunk")


class GroundedAnswer(BaseModel):
    """The final product of the RAG pipeline: an answer with receipts.

    `grounded=False` means the answer failed (or could not complete) the
    faithfulness self-check — callers must present it with a warning or
    escalate to a human rather than stating it as fact. Token counts ride
    along because cost-per-answer is an operational metric, not an
    afterthought (formalized in phase 8).
    """

    question: str
    answer: str
    citations: list[Citation]
    grounded: bool
    notes: str | None = Field(
        default=None, description="Self-check diagnostics, e.g. unsupported claims"
    )
    input_tokens: int = 0
    output_tokens: int = 0
