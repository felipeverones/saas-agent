"""Human-in-the-loop approval — the graph learns to WAIT.

HOW interrupt() WORKS (and why checkpointing was the prerequisite)
When this node calls `interrupt(payload)`, LangGraph raises internally,
persists the whole graph state at this exact point, and RETURNS control to
the caller with the payload surfaced. The process can exit. Hours or days
later — same or different process — invoking the same thread with
`Command(resume=decision)` re-enters HERE, with `decision` as interrupt()'s
return value, and the flow continues as if it never stopped.

This is the payoff of choosing a checkpointed state graph in phase 0: a
plain function call stack cannot wait for a human without holding a process
(and a prayer) open. A checkpoint can wait forever.

THE SAFETY INVARIANT: code, not the model, enforces the stop. The LLM already
proposed the refund; whether money moves is decided strictly on this side of
the interrupt — an injected or confused model cannot cross it. Denial is also
fail-closed: an ambiguous resume payload counts as "denied".
"""

from langgraph.types import interrupt

from nimbusdesk.agents.state import SupportState


def human_approval_node(state: SupportState) -> dict:
    refund = state.pending_refund
    assert refund is not None, "approval node reached without a pending refund"

    # Everything a human needs to decide, as data. This payload is what the
    # operator UI (or our CLI prompt) renders.
    decision = interrupt(
        {
            "action": "issue_refund",
            "email": refund.email,
            "amount_usd": refund.amount_usd,
            "reason": refund.reason,
            "customer_question": state.question,
        }
    )

    approved = isinstance(decision, dict) and decision.get("approved") is True
    note = decision.get("note", "") if isinstance(decision, dict) else ""

    if approved:
        return {
            "refund_decision": "approved",
            "resolved_by": "billing+human",
            "final_answer": (
                f"Good news — your refund of ${refund.amount_usd:.2f} was approved "
                "by our billing team and will reach your original payment method "
                "within 5-10 business days."
            ),
        }
    return {
        "refund_decision": "denied",
        "resolved_by": "billing+human",
        "escalated": True,
        "escalation_reason": f"refund denied by human reviewer{': ' + note if note else ''}",
        "final_answer": (
            f"Your refund request of ${refund.amount_usd:.2f} was reviewed by our "
            "billing team and could not be approved"
            + (f" ({note})" if note else "")
            + ". A specialist will follow up with alternatives."
        ),
    }
