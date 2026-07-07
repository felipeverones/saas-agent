# Standardized commands so nobody has to remember tool invocations.
# WHY a Makefile in 2026: it's still the lingua franca for "how do I run this repo" —
# CI, Docker builds and new contributors all read the same entry points.

.PHONY: setup test lint run cli compose eval ingest search ask agent team chat mcp-crm mcp-ticketing up down

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

ingest:           ## Index data/seed into Qdrant (idempotent — safe to re-run)
	uv run python -m nimbusdesk.rag ingest

search:           ## Debug retrieval (no LLM), e.g.: make search Q="refund policy"
	uv run python -m nimbusdesk.rag search "$(Q)"

ask:              ## Grounded answer with citations (needs API key in .env)
	uv run python -m nimbusdesk.rag ask "$(Q)"

agent:            ## Single support agent with tools (needs API key in .env)
	uv run python -m nimbusdesk.agents solo "$(Q)"

team:             ## Multi-agent graph: triage + supervisor + specialists
	uv run python -m nimbusdesk.agents team "$(Q)"

chat:             ## Interactive multi-turn chat with memory, e.g.: make chat EMAIL=dana@acme.io
	uv run python -m nimbusdesk.agents chat --email "$(EMAIL)" --thread "$(or $(THREAD),chat)"

mcp-crm:          ## Run the CRM MCP server (http://localhost:8101/mcp)
	uv run python -m nimbusdesk.mcp_servers.crm

mcp-ticketing:    ## Run the ticketing MCP server (http://localhost:8102/mcp)
	uv run python -m nimbusdesk.mcp_servers.ticketing

run:              ## Start the API from the venv (dev mode; needs `make up` first)
	uv run python -m nimbusdesk.interface.api

cli:              ## Interactive chat client against the running API
	uv run nimbus chat --email "$(EMAIL)"

compose:          ## The whole system in containers: qdrant + phoenix + api
	docker compose up --build

eval:             ## Golden-dataset evals (retrieval is free; routing/faithfulness need API key)
	uv run python evals/run_eval.py
