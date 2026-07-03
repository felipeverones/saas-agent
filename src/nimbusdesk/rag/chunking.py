"""Chunking — splitting documents into retrieval-sized pieces.

WHY CHUNK AT ALL
We embed and retrieve CHUNKS, not whole documents, because (a) an embedding of
a 3-page article averages many topics into one blurry vector, and (b) the LLM
context window should receive only the relevant slice, not the whole file.

THE TRADE-OFF EVERY CHUNKER NAVIGATES
- Too small: a chunk loses the context needed to understand it ("it expires
  after 14 days" — what does?).
- Too big: the vector gets blurry and retrieval precision drops; irrelevant
  text rides along into the prompt.

OUR STRATEGY (structure-aware, a.k.a. "semantic-ish")
1. Split on markdown headings — authors already segmented the document by
   topic; throwing that structure away (fixed-size splitting) is the classic
   tutorial mistake.
2. Only if a section exceeds `max_chars`, split it further on paragraph
   boundaries, carrying the previous paragraph as OVERLAP so a sentence's
   context isn't severed at an arbitrary cut point.
3. Each chunk keeps its heading trail in `section` — used both to enrich the
   embedded text (see ingestion.py) and for human-readable citations.

Chunk ids are deterministic (UUID5 of doc_id + position): re-ingesting the
same corpus UPSERTS the same points instead of duplicating them.
"""

import re
import uuid

from nimbusdesk.domain.knowledge import DocumentChunk, SourceDocument

MAX_CHUNK_CHARS = 1200
OVERLAP_CHARS = 200

_HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$")


def chunk_document(
    doc: SourceDocument,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for section_trail, section_text in _split_by_headings(doc.text):
        for piece in _split_long_section(section_text, max_chars, overlap_chars):
            position = len(chunks)
            chunks.append(
                DocumentChunk(
                    chunk_id=make_chunk_id(doc.doc_id, position),
                    doc_id=doc.doc_id,
                    title=doc.title,
                    section=section_trail,
                    text=piece,
                    position=position,
                )
            )
    return chunks


def make_chunk_id(doc_id: str, position: int) -> str:
    """Deterministic id: same document + position -> same UUID, every run.

    UUID5 (name-based) rather than UUID4 (random) is what makes re-ingestion
    idempotent — and Qdrant requires point ids to be UUIDs or integers.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"nimbusdesk:{doc_id}:{position}"))


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Yield (heading trail, body) per section. The trail concatenates the
    active h1/h2/h3 ("Billing > Refund policy"), so nested context survives."""
    sections: list[tuple[str, str]] = []
    trail: dict[int, str] = {}
    body_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(body_lines).strip()
        if body:
            sections.append((" > ".join(trail[lvl] for lvl in sorted(trail)), body))

    for line in text.splitlines():
        match = _HEADING_PATTERN.match(line)
        if match:
            flush()
            body_lines.clear()
            level = len(match.group(1))
            trail[level] = match.group(2).strip()
            for deeper in range(level + 1, 4):  # a new h2 resets any old h3
                trail.pop(deeper, None)
        else:
            body_lines.append(line)
    flush()
    return sections


def _split_long_section(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split an oversized section on paragraph boundaries.

    Overlap strategy: each new piece starts with the previous piece's last
    paragraph (when it's small enough) so no sentence loses its neighbor.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            pieces.append(current)
            tail = current.split("\n\n")[-1]
            current = f"{tail}\n\n{para}" if len(tail) <= overlap_chars else para
        else:
            current = para

        # Degenerate case: one paragraph alone exceeds max_chars (huge tables,
        # code dumps). Hard-split it — ugly, but bounded memory beats elegance.
        while len(current) > max_chars:
            pieces.append(current[:max_chars])
            current = current[max_chars - overlap_chars :]

    if current:
        pieces.append(current)
    return pieces
