from pathlib import Path

import pytest

from nimbusdesk.rag.loading import load_markdown_dir


def test_loads_docs_with_title_from_h1(tmp_path: Path):
    (tmp_path / "refund-policy.md").write_text(
        "# Refunds made simple\n\nBody text.", encoding="utf-8"
    )
    (tmp_path / "no-heading.md").write_text("Just text, no heading.", encoding="utf-8")

    docs = {d.doc_id: d for d in load_markdown_dir(tmp_path)}

    assert docs["refund-policy"].title == "Refunds made simple"
    # Fallback: filename stem, de-slugged — a doc must never have an empty title
    assert docs["no-heading"].title == "no heading"


def test_empty_directory_raises(tmp_path: Path):
    # Silent empty ingestion is a production incident (an index quietly wiped
    # or pointed at the wrong path) — fail loudly instead.
    with pytest.raises(FileNotFoundError):
        load_markdown_dir(tmp_path)
