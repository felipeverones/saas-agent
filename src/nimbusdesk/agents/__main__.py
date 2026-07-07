"""Developer entry point for the agents.

    uv run python -m nimbusdesk.agents solo "question"     # phase 3: one agent
    uv run python -m nimbusdesk.agents team "question"     # phase 4: full graph
        [--email dana@acme.io] [--thread ticket-123]

`--thread` demonstrates checkpointing: runs with the same thread id share
persisted state in data/checkpoints.sqlite (the graph's short-term memory).
"""

import argparse
import sqlite3
import sys

from qdrant_client import QdrantClient

from nimbusdesk.agents.support_agent import build_support_agent
from nimbusdesk.infrastructure.anthropic_llm import AnthropicProvider, MissingApiKeyError
from nimbusdesk.infrastructure.embeddings import FastEmbedEmbedder, FastEmbedSparseEmbedder
from nimbusdesk.infrastructure.reranker import FastEmbedReranker
from nimbusdesk.infrastructure.settings import get_settings
from nimbusdesk.infrastructure.vector_store import QdrantVectorIndex
from nimbusdesk.llm.tracking import UsageTracker
from nimbusdesk.rag.retrieval import Retriever

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

CHECKPOINT_DB = "data/checkpoints.sqlite"


def _build_retrieval(settings) -> tuple[Retriever, FastEmbedReranker]:
    retriever = Retriever(
        FastEmbedEmbedder(settings.embedding_model_name, settings.embedding_dimension),
        FastEmbedSparseEmbedder(settings.sparse_model_name),
        QdrantVectorIndex(
            client=QdrantClient(url=settings.qdrant_url),
            collection=settings.qdrant_collection,
            dimension=settings.embedding_dimension,
        ),
    )
    return retriever, FastEmbedReranker(settings.reranker_model_name)


def _build_llms(settings) -> tuple[UsageTracker, UsageTracker]:
    api_key = (
        settings.anthropic_api_key.get_secret_value()
        if settings.anthropic_api_key
        else None
    )
    fast = UsageTracker(AnthropicProvider(api_key=api_key, model=settings.nimbus_model_fast))
    strong = UsageTracker(
        AnthropicProvider(api_key=api_key, model=settings.nimbus_model_strong)
    )
    return fast, strong


def _load_mcp_tools(settings):
    """Discover tools from both MCP servers (they must be running — see
    `make mcp-crm` / `make mcp-ticketing`). Sensitive tools go through the
    CLI consent prompt; read-only ones run freely."""
    from nimbusdesk.mcp_clients.client import SyncMcpClient
    from nimbusdesk.mcp_clients.tools import CliConsent, load_remote_tools

    consent = CliConsent()
    tools = []
    for url in (settings.mcp_crm_url, settings.mcp_ticketing_url):
        try:
            tools.extend(load_remote_tools(SyncMcpClient(url), consent))
        except Exception as error:
            print(
                f"error: could not reach MCP server at {url} "
                f"({type(error).__name__}). Start it first: make mcp-crm / "
                "make mcp-ticketing",
                file=sys.stderr,
            )
            raise SystemExit(1) from None
    print(f"[mcp] loaded {len(tools)} remote tools: {', '.join(t.name for t in tools)}")
    return tools


def _cmd_solo(question: str, use_mcp: bool) -> None:
    settings = get_settings()
    _, strong = _build_llms(settings)
    retriever, reranker = _build_retrieval(settings)
    account_tools = _load_mcp_tools(settings) if use_mcp else None
    agent = build_support_agent(strong, retriever, reranker, account_tools=account_tools)

    result = agent.run(question)

    if result.steps:
        print("--- agent steps " + "-" * 44)
        for i, step in enumerate(result.steps, start=1):
            flag = " [error]" if step.is_error else ""
            print(f"{i}. {step.tool}({step.arguments}){flag}")
            print(f"   -> {step.observation[:160]}".replace("\n", " "))
        print("-" * 60)
    print(f"\n{result.answer}\n")
    limit = " | HIT ITERATION LIMIT" if result.hit_iteration_limit else ""
    print(
        f"({result.iterations} iteration(s), {len(result.steps)} tool call(s), "
        f"tokens: {strong.input_tokens} in / {strong.output_tokens} out{limit})"
    )


def _build_memory(settings, fast_llm):
    """Long-term memory wiring: profile facts in SQLite, episodes in Qdrant
    (same instance as the KB, separate collection), extraction on the fast
    model tier."""
    from nimbusdesk.memory.episodic import EpisodicMemoryStore
    from nimbusdesk.memory.profile_store import SqliteProfileStore
    from nimbusdesk.memory.service import MemoryService
    from nimbusdesk.memory.writer import MemoryWriter

    profiles = SqliteProfileStore(
        sqlite3.connect(settings.memory_db_path, check_same_thread=False)
    )
    episodes = EpisodicMemoryStore(
        client=QdrantClient(url=settings.qdrant_url),
        embedder=FastEmbedEmbedder(
            settings.embedding_model_name, settings.embedding_dimension
        ),
        collection=settings.memory_collection,
    )
    return MemoryService(profiles, episodes, MemoryWriter(fast_llm, profiles, episodes))


def _build_team_graph(settings, use_mcp: bool):
    from langgraph.checkpoint.sqlite import SqliteSaver

    from nimbusdesk.agents.graph import build_support_graph

    fast, strong = _build_llms(settings)
    retriever, reranker = _build_retrieval(settings)
    account_tools = _load_mcp_tools(settings) if use_mcp else None

    # check_same_thread=False: LangGraph may touch the connection from worker
    # threads; SQLite forbids cross-thread use by default.
    checkpointer = SqliteSaver(sqlite3.connect(CHECKPOINT_DB, check_same_thread=False))
    graph = build_support_graph(
        fast,
        strong,
        retriever,
        reranker,
        checkpointer,
        account_tools=account_tools,
        memory=_build_memory(settings, fast),
    )
    return graph, fast, strong


def _cli_approval(payload: dict) -> dict:
    """The human side of the interrupt: render the pending action, ask.
    In production this payload would land in an operator queue instead."""
    print("\n=== HUMAN APPROVAL REQUIRED " + "=" * 32)
    for key, value in payload.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    reply = input("approve this action? [y/N] ").strip().lower()
    if reply in ("y", "yes"):
        return {"approved": True}
    note = input("optional note for the customer: ").strip()
    return {"approved": False, "note": note}


def _cmd_team(question: str, email: str | None, thread: str, use_mcp: bool) -> None:
    from nimbusdesk.agents.graph import run_support_graph

    settings = get_settings()
    graph, fast, strong = _build_team_graph(settings, use_mcp)

    state = run_support_graph(
        graph, question, customer_email=email, thread_id=thread,
        approval_callback=_cli_approval,
    )

    if state.triage:
        print(
            f"triage: {state.triage.category} / {state.triage.priority} "
            f"(confidence {state.triage.confidence:.2f}) — {state.triage.summary}"
        )
    print(f"resolved by: {state.resolved_by}")
    if state.escalated:
        print(f"ESCALATED: {state.escalation_reason}")
    if state.failures:
        print(f"failures recorded: {state.failures}")
    print(f"\n{state.final_answer}\n")
    print(
        f"(thread={thread} | supervisor visits: {state.supervisor_visits} | tokens: "
        f"{fast.input_tokens + strong.input_tokens} in / "
        f"{fast.output_tokens + strong.output_tokens} out)"
    )


def _cmd_chat(email: str | None, thread: str, use_mcp: bool) -> None:
    """Interactive multi-turn chat — short-term memory (history within this
    thread) and long-term memory (recall across sessions) both live here."""
    from nimbusdesk.agents.graph import run_support_graph

    settings = get_settings()
    graph, fast, strong = _build_team_graph(settings, use_mcp)

    print(f"NimbusDesk support chat — thread '{thread}'"
          + (f", customer {email}" if email else ""))
    print("Type your message ('exit' to quit).\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("exit", "quit", "sair"):
            break
        state = run_support_graph(
            graph, question, customer_email=email, thread_id=thread,
            approval_callback=_cli_approval,
        )
        tag = state.resolved_by or "?"
        print(f"\nnimbus[{tag}]> {state.final_answer}\n")
    total_in = fast.input_tokens + strong.input_tokens
    total_out = fast.output_tokens + strong.output_tokens
    print(f"\n(session tokens: {total_in} in / {total_out} out)")


def main() -> None:
    parser = argparse.ArgumentParser(prog="nimbusdesk.agents")
    sub = parser.add_subparsers(dest="command", required=True)

    solo = sub.add_parser("solo", help="Single support agent (phase 3)")
    solo.add_argument("question")
    solo.add_argument("--mcp", action="store_true", help="use MCP servers for CRM/ticketing")

    team = sub.add_parser("team", help="Supervisor + specialists graph (phase 4)")
    team.add_argument("question")
    team.add_argument("--email", default=None)
    team.add_argument("--thread", default="dev")
    team.add_argument("--mcp", action="store_true", help="use MCP servers for CRM/ticketing")

    chat = sub.add_parser("chat", help="Interactive multi-turn chat with memory (phase 6)")
    chat.add_argument("--email", default=None)
    chat.add_argument("--thread", default="chat")
    chat.add_argument("--mcp", action="store_true", help="use MCP servers for CRM/ticketing")

    args = parser.parse_args()

    from anthropic import AuthenticationError

    try:
        if args.command == "solo":
            _cmd_solo(args.question, args.mcp)
        elif args.command == "team":
            _cmd_team(args.question, args.email, args.thread, args.mcp)
        else:
            _cmd_chat(args.email, args.thread, args.mcp)
    except MissingApiKeyError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    except AuthenticationError:
        print(
            "error: Anthropic rejected the API key (401). Check ANTHROPIC_API_KEY "
            "in your .env — it may be a placeholder, revoked, or truncated.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
