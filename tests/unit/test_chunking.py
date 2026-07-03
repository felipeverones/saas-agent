"""Unit tests for the structure-aware chunker.

Chunking bugs are silent: nothing crashes, retrieval just quietly degrades.
These tests pin the properties that retrieval quality depends on.
"""

from nimbusdesk.domain.knowledge import SourceDocument
from nimbusdesk.rag.chunking import chunk_document, make_chunk_id

DOC = SourceDocument(
    doc_id="refunds",
    title="Billing and refunds",
    path="refunds.md",
    text=(
        "# Billing and refunds\n\n"
        "Intro paragraph about billing.\n\n"
        "## Refund policy\n\n"
        "Monthly plans are refundable within 14 days.\n\n"
        "### Enterprise exceptions\n\n"
        "Enterprise contracts define their own terms.\n\n"
        "## Payment methods\n\n"
        "We accept credit cards and ACH.\n"
    ),
)


def test_sections_follow_headings():
    chunks = chunk_document(DOC)
    sections = [c.section for c in chunks]
    assert sections == [
        "Billing and refunds",
        "Billing and refunds > Refund policy",
        "Billing and refunds > Refund policy > Enterprise exceptions",
        "Billing and refunds > Payment methods",
    ]


def test_h2_resets_previous_h3_in_trail():
    chunks = chunk_document(DOC)
    last = chunks[-1].section
    assert "Enterprise exceptions" not in last, "stale h3 leaked into a later h2's trail"


def test_positions_are_sequential_and_ids_deterministic():
    first = chunk_document(DOC)
    second = chunk_document(DOC)
    assert [c.position for c in first] == list(range(len(first)))
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first), "ids must be unique"
    assert first[0].chunk_id == make_chunk_id("refunds", 0)


def test_long_sections_split_within_limit_and_overlap():
    paragraphs = [f"Paragraph {i}: " + ("support ticket detail. " * 12) for i in range(8)]
    doc = SourceDocument(
        doc_id="long",
        title="Long doc",
        path="long.md",
        text="# Long doc\n\n## Big section\n\n" + "\n\n".join(paragraphs),
    )
    chunks = chunk_document(doc, max_chars=700, overlap_chars=300)

    assert len(chunks) > 1, "oversized section should split"
    assert all(len(c.text) <= 700 for c in chunks)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        last_paragraph = previous.text.split("\n\n")[-1]
        assert current.text.startswith(last_paragraph), (
            "each piece should start with the previous piece's last paragraph (overlap)"
        )


def test_giant_single_paragraph_is_hard_split():
    doc = SourceDocument(
        doc_id="giant",
        title="Giant",
        path="giant.md",
        text="# Giant\n\n" + "x" * 5000,
    )
    chunks = chunk_document(doc, max_chars=1000, overlap_chars=100)
    assert all(len(c.text) <= 1000 for c in chunks)
    assert sum(len(c.text) for c in chunks) >= 5000, "no content may be dropped"
