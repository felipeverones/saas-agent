"""crm-mcp-server — mocked CRM exposed over the Model Context Protocol.

PROTOCOL NOTES
- Built on the official `mcp` SDK (FastMCP API), which negotiates the protocol
  revision with each client at initialize time; with SDK 1.28.x that is the
  2025-06-18 spec revision — the one that made Streamable HTTP the standard
  transport (the older SSE transport is deprecated; see ADR-06).
- Tool input schemas are GENERATED from the Python type hints below and
  ENFORCED server-side by FastMCP before our functions run. A hostile or
  confused client cannot reach our code with malformed arguments — schema
  validation at the protocol boundary is the baseline defense.
- Tools carry ANNOTATIONS (readOnlyHint/destructiveHint): machine-readable
  hints that let clients apply the MCP consent model — our client requires
  explicit user approval before any non-read-only tool runs (mcp_clients/).

DELIBERATE ISOLATION: this package imports nothing from the rest of
nimbusdesk. It plays the role of a system owned by another team — if it
imported our agents, the decoupling MCP exists to provide would be theater.
"""

import json
import os
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

mcp = FastMCP(
    "nimbus-crm",
    host="127.0.0.1",
    port=int(os.environ.get("MCP_CRM_PORT", "8101")),
)

# In-memory fake data: resets on restart, which is fine for a demo system.
# (Same personas as the phase-3 local tools on purpose — the point of phase 5
# is moving the same capability behind the protocol.)
_CUSTOMERS: dict[str, dict] = {
    "dana@acme.io": {
        "name": "Dana Reyes",
        "company": "Acme Corp",
        "plan": "business",
        "seats": 140,
        "status": "active",
        "payment_state": "current",
    },
    "li.wei@bitforge.dev": {
        "name": "Li Wei",
        "company": "BitForge",
        "plan": "pro",
        "seats": 12,
        "status": "active",
        "payment_state": "past_due",
    },
    "sam@nimbusfan.com": {
        "name": "Sam Ortiz",
        "company": "—",
        "plan": "free",
        "seats": 3,
        "status": "active",
        "payment_state": "current",
    },
}

Email = Annotated[str, Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")]


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def lookup_customer(email: Email) -> str:
    """Look up a customer's account by email: plan, seats, subscription and
    payment status."""
    customer = _CUSTOMERS.get(email.lower())
    if customer is None:
        return f"no customer found with email {email}"
    return json.dumps(customer)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
def update_customer_plan(
    email: Email,
    new_plan: Literal["free", "pro", "business", "enterprise"],
) -> str:
    """Change a customer's subscription plan. SENSITIVE: modifies customer
    data — clients must obtain user consent before calling."""
    customer = _CUSTOMERS.get(email.lower())
    if customer is None:
        return f"no customer found with email {email}"
    old_plan = customer["plan"]
    customer["plan"] = new_plan
    return json.dumps({"email": email, "previous_plan": old_plan, "new_plan": new_plan})


@mcp.resource("crm://customers")
def customer_directory() -> str:
    """Directory of known customer emails (a RESOURCE: readable context data,
    as opposed to a TOOL, which is an action the model invokes)."""
    return json.dumps(sorted(_CUSTOMERS.keys()))
