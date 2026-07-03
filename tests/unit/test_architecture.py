"""Architecture tests — the dependency rule as executable code.

CONCEPT (fitness functions)
Instead of trusting code review to keep the domain layer pure, we make the
architecture a test: parse every file in `domain/` and fail if it imports an AI
framework, a vendor SDK, or any other nimbusdesk layer. This is called an
"architectural fitness function" — convention decays under deadline pressure,
CI doesn't.

We use `ast` (parsing source text) rather than importing modules, so the test
works even on modules with side effects and needs no dependencies installed.
"""

import ast
from pathlib import Path

SRC = Path(__file__).parents[2] / "src" / "nimbusdesk"
DOMAIN = SRC / "domain"

# The domain layer may use the stdlib and pydantic. Everything below is banned:
# vendors, frameworks, and infra concerns that would weld business rules to tech
# choices we want to be able to swap.
FORBIDDEN_IN_DOMAIN = {
    "anthropic",
    "openai",
    "langgraph",
    "langchain",
    "langchain_core",
    "mcp",
    "qdrant_client",
    "fastembed",
    "fastapi",
    "httpx",
    "requests",
    "sqlite3",
}


def _imported_top_level_names(py_file: Path) -> set[str]:
    """Return the top-level package name of every import in a file."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def _domain_files() -> list[Path]:
    files = list(DOMAIN.rglob("*.py"))
    assert files, "domain package not found — did the layout change?"
    return files


def test_domain_imports_no_vendor_or_framework():
    """domain/ must stay pure: no AI SDKs, no frameworks, no I/O libraries."""
    for py_file in _domain_files():
        offending = _imported_top_level_names(py_file) & FORBIDDEN_IN_DOMAIN
        assert not offending, (
            f"{py_file.relative_to(SRC)} imports {sorted(offending)} — "
            "the domain layer must not depend on vendors/frameworks (see ADR-13)"
        )


def _imported_module_paths(py_file: Path) -> set[str]:
    """Return full dotted module paths of absolute imports (relative ones are
    internal to the package by definition, so they're allowed)."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    paths: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            paths.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            paths.add(node.module)
    return paths


def test_domain_imports_no_other_nimbusdesk_layer():
    """domain/ is the innermost circle: it may not import sibling packages
    (agents, rag, infrastructure, ...). Imports within nimbusdesk.domain are fine."""
    for py_file in _domain_files():
        for mod in _imported_module_paths(py_file):
            if mod == "nimbusdesk" or mod.startswith("nimbusdesk."):
                assert mod.startswith("nimbusdesk.domain"), (
                    f"{py_file.relative_to(SRC)} imports '{mod}' — "
                    "domain must not depend on outer layers (dependency rule, ADR-13)"
                )
