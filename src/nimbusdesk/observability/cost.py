"""Token -> USD estimation. Rough by design, load-bearing in practice.

Operating agents means watching cost-per-resolved-ticket like a product
metric: an agent that burns $3 of tokens on a $0.50 question is a bug even
when the answer is right. Estimates below use published per-million-token
prices; ADJUST THE TABLE when models/prices change — the point is the ORDER
OF MAGNITUDE per run being visible on every CLI output, not cent accuracy.
"""

# (input USD, output USD) per MILLION tokens — estimates, see docstring.
MODEL_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}
# Unknown model: assume strong-tier pricing — overestimating beats
# underestimating when the number exists to catch runaway cost.
DEFAULT_PRICES = (3.00, 15.00)


def estimate_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = MODEL_PRICES_PER_MTOK.get(model, DEFAULT_PRICES)
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def format_usage(pairs: list[tuple[str, int, int]]) -> str:
    """pairs = [(model, input_tokens, output_tokens), ...] -> one CLI line."""
    total_in = sum(p[1] for p in pairs)
    total_out = sum(p[2] for p in pairs)
    total_usd = sum(estimate_usd(*p) for p in pairs)
    return f"tokens: {total_in} in / {total_out} out | est. cost: ${total_usd:.4f}"
