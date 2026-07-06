"""ticketing-mcp-server — CRUD over fake support tickets, exposed via MCP.

Same protocol notes as the CRM server (see crm/server.py): official SDK,
Streamable HTTP, schemas generated+enforced from type hints, consent
annotations on every mutating tool. Write tools here (create_ticket,
update_ticket_status) are the ones that exercise the consent gate — and in
phase 7, the human-in-the-loop flow.
"""

import json
import os
from itertools import count
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

mcp = FastMCP(
    "nimbus-ticketing",
    host="127.0.0.1",
    port=int(os.environ.get("MCP_TICKETING_PORT", "8102")),
)

TicketStatus = Literal["open", "in_progress", "waiting_customer", "resolved", "closed"]

# Seeded in-memory store (resets on restart — fine for a demo system).
_TICKETS: dict[str, dict] = {
    "TKT-1001": {
        "id": "TKT-1001",
        "email": "dana@acme.io",
        "subject": "Sync stuck at 99% on several machines",
        "body": "Since yesterday about 20 of our users report sync stuck at 99%.",
        "priority": "high",
        "status": "in_progress",
    },
    "TKT-1002": {
        "id": "TKT-1002",
        "email": "li.wei@bitforge.dev",
        "subject": "Invoice for May missing",
        "body": "I can't find our May invoice in the billing page.",
        "priority": "normal",
        "status": "open",
    },
}
_NEXT_ID = count(1003)

Email = Annotated[str, Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")]


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_ticket(ticket_id: str) -> str:
    """Fetch one ticket by id (e.g. TKT-1001)."""
    ticket = _TICKETS.get(ticket_id.upper())
    return json.dumps(ticket) if ticket else f"no ticket found with id {ticket_id}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def list_tickets(email: Email | None = None) -> str:
    """List tickets, optionally filtered by customer email."""
    tickets = [
        t for t in _TICKETS.values() if email is None or t["email"] == email.lower()
    ]
    return json.dumps(tickets)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
def create_ticket(
    email: Email,
    subject: Annotated[str, Field(min_length=3, max_length=200)],
    body: Annotated[str, Field(min_length=3, max_length=5000)],
    priority: Literal["low", "normal", "high", "urgent"] = "normal",
) -> str:
    """Open a new support ticket. SENSITIVE: creates a record on the
    customer's account — clients must obtain user consent before calling."""
    ticket_id = f"TKT-{next(_NEXT_ID)}"
    _TICKETS[ticket_id] = {
        "id": ticket_id,
        "email": email.lower(),
        "subject": subject,
        "body": body,
        "priority": priority,
        "status": "open",
    }
    return json.dumps(_TICKETS[ticket_id])


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
def update_ticket_status(ticket_id: str, status: TicketStatus) -> str:
    """Change a ticket's status. SENSITIVE (destructive hint: closing a ticket
    ends the conversation for the customer) — consent required."""
    ticket = _TICKETS.get(ticket_id.upper())
    if ticket is None:
        return f"no ticket found with id {ticket_id}"
    previous = ticket["status"]
    ticket["status"] = status
    return json.dumps({"id": ticket["id"], "previous_status": previous, "status": status})


@mcp.resource("tickets://open")
def open_tickets() -> str:
    """All currently open/in-progress tickets (resource: context, not action)."""
    active = [t for t in _TICKETS.values() if t["status"] in ("open", "in_progress")]
    return json.dumps(active)
