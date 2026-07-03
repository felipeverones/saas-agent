# Standardized commands so nobody has to remember tool invocations.
# WHY a Makefile in 2026: it's still the lingua franca for "how do I run this repo" —
# CI, Docker builds and new contributors all read the same entry points.

.PHONY: setup test lint run eval ingest up down

setup:            ## Create venv + install all deps (uv provisions Python 3.12 itself)
	uv sync

test:             ## Run the full test suite (LLM calls are always mocked)
	uv run pytest

lint:             ## Static checks (style + common bug patterns)
	uv run ruff check src tests

up:               ## Start local infra (Qdrant vector store + Phoenix trace viewer)
	docker compose up -d

down:             ## Stop local infra
	docker compose down

ingest:           ## Load seed documents into the vector store (available in phase 1)
	@echo "Available in phase 1 (RAG ingestion)"

run:              ## Start the API + interactive CLI (available in phase 9)
	@echo "Available in phase 9 (packaging & interface)"

eval:             ## Run the golden-dataset evaluation suite (available in phase 8)
	@echo "Available in phase 8 (observability & evaluation)"
