"""Document loading — the first step of the ingestion pipeline.

Deliberately boring: read markdown files, extract a title, produce
`SourceDocument`s. In a real deployment this step is where connectors live
(Confluence, Notion, Zendesk exports...) — which is exactly why it's isolated
in its own module: new sources should never touch chunking or indexing.
"""

import re
from pathlib import Path

from nimbusdesk.domain.knowledge import SourceDocument

_H1_PATTERN = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def load_markdown_dir(directory: Path) -> list[SourceDocument]:
    """Load every .md file in `directory` (non-recursive) as a SourceDocument.

    doc_id is the filename stem — stable across runs, which downstream chunk
    ids (and therefore idempotent re-ingestion) depend on.
    """
    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No .md documents found in {directory}")

    documents = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        match = _H1_PATTERN.search(text)
        title = match.group(1).strip() if match else path.stem.replace("-", " ")
        documents.append(
            SourceDocument(doc_id=path.stem, title=title, path=str(path), text=text)
        )
    return documents
