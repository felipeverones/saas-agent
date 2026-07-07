# Multi-stage build with uv: deps resolve against the LOCKFILE (reproducible),
# and the final image carries the venv + source, not the build tooling.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app

# Layer-cache trick: dependencies change rarely, source changes often.
# Installing deps from the lockfile FIRST means code edits don't re-download
# the world on every build.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY README.md ./
RUN uv sync --frozen --no-dev

FROM python:3.12-slim-bookworm
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY data/seed ./data/seed
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    # Embedding/reranker models (~90MB) download here on first start; the
    # compose file mounts a volume so restarts don't re-download.
    FASTEMBED_CACHE_PATH=/models

EXPOSE 8000
CMD ["python", "-m", "nimbusdesk.interface.api"]
