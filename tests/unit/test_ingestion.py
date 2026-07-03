"""Pipeline mechanics, tested end to end against fakes — no ML, no I/O."""

from pathlib import Path

from nimbusdesk.rag.ingestion import IngestionPipeline
from tests.fakes import FakeEmbedder, InMemoryVectorIndex


def _write_corpus(tmp_path: Path) -> Path:
    (tmp_path / "a.md").write_text("# Doc A\n\n## S1\n\nAlpha text.", encoding="utf-8")
    (tmp_path / "b.md").write_text(
        "# Doc B\n\n## S1\n\nBeta text.\n\n## S2\n\nGamma text.", encoding="utf-8"
    )
    return tmp_path


def test_report_counts_documents_and_chunks(tmp_path: Path):
    index = InMemoryVectorIndex()
    report = IngestionPipeline(FakeEmbedder(), index).run(_write_corpus(tmp_path))

    assert report.documents == 2
    assert report.chunks == 3
    assert len(index) == 3


def test_reingestion_is_idempotent(tmp_path: Path):
    """Deterministic chunk ids + upsert = running ingest twice must not
    duplicate points. This is what makes `make ingest` safe to re-run."""
    corpus = _write_corpus(tmp_path)
    index = InMemoryVectorIndex()
    pipeline = IngestionPipeline(FakeEmbedder(), index)

    pipeline.run(corpus)
    size_after_first = len(index)
    pipeline.run(corpus)

    assert len(index) == size_after_first
