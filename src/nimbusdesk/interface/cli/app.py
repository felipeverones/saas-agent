"""nimbus — the interactive support chat client.

    uv run nimbus chat [--email dana@acme.io] [--thread t1] [--api-url ...]

DESIGN: this CLI holds ZERO business logic — it is an SSE-consuming HTTP
client of the API, which means every demo through it also exercises the
production surface (one brain, two doors; see interface/__init__.py).
It even plays the operator role: when the stream pauses with
`approval_required`, it prompts the human and POSTs the verdict to
/approvals/{thread}, resuming the checkpointed run.
"""

import json
import sys

import httpx
import typer

cli = typer.Typer(add_completion=False, help="NimbusDesk support chat client")

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


@cli.command()
def chat(
    email: str = typer.Option(None, help="Customer email (enables memory + account tools)"),
    thread: str = typer.Option("cli", help="Conversation thread id"),
    api_url: str = typer.Option("http://localhost:8000", help="NimbusDesk API base URL"),
) -> None:
    """Interactive multi-turn chat against the running API."""
    client = httpx.Client(base_url=api_url, timeout=180.0)
    _check_health(client, api_url)

    typer.echo(f"NimbusDesk chat — thread '{thread}'" + (f", {email}" if email else ""))
    typer.echo("Type your message ('exit' to quit).\n")
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not message or message.lower() in ("exit", "quit", "sair"):
            break
        _run_turn(client, message, email, thread)


def _check_health(client: httpx.Client, api_url: str) -> None:
    try:
        health = client.get("/health").json()
    except httpx.HTTPError:
        typer.secho(f"error: API not reachable at {api_url} — run `make run` first",
                    fg="red")
        raise typer.Exit(1) from None
    if health.get("status") != "ok":
        typer.secho(f"API is degraded: {health.get('detail')}", fg="yellow")


def _run_turn(client: httpx.Client, message: str, email: str | None, thread: str) -> None:
    body = {"message": message, "email": email, "thread_id": thread}
    with client.stream("POST", "/chat", json=body) as response:
        for event, data in _iter_sse(response):
            if event == "node":
                typer.secho(f"  … {data['node']}", fg="bright_black")
            elif event == "answer":
                _print_answer(data)
            elif event == "approval_required":
                _handle_approval(client, data)
            elif event == "error":
                typer.secho(f"error: {data['detail']}", fg="red")


def _handle_approval(client: httpx.Client, payload: dict) -> None:
    """The CLI wearing the operator hat: render the pending action, decide,
    resume the checkpointed run via the approvals endpoint."""
    typer.secho("\n=== HUMAN APPROVAL REQUIRED " + "=" * 24, fg="yellow")
    for key, value in payload.items():
        if key != "thread_id":
            typer.echo(f"  {key}: {value}")
    approved = input("approve this action? [y/N] ").strip().lower() in ("y", "yes")
    note = "" if approved else input("optional note for the customer: ").strip()

    result = client.post(
        f"/approvals/{payload['thread_id']}", json={"approved": approved, "note": note}
    )
    result.raise_for_status()
    _print_answer(result.json())


def _print_answer(data: dict) -> None:
    tag = data.get("resolved_by") or "?"
    typer.echo(f"\nnimbus[{tag}]> {data['answer']}\n")
    if data.get("escalated"):
        typer.secho("  (escalated to a human specialist)", fg="yellow")
    typer.secho(f"  (session est. cost: ${data.get('session_est_cost_usd', 0):.4f})",
                fg="bright_black")


def _iter_sse(response: httpx.Response):
    """Minimal SSE parser: yields (event, parsed_json_data) pairs."""
    event = None
    for line in response.iter_lines():
        if line.startswith("event: "):
            event = line.removeprefix("event: ").strip()
        elif line.startswith("data: ") and event:
            yield event, json.loads(line.removeprefix("data: "))
            event = None


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
