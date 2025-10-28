"""Utility helpers to estimate infrastructure costs for LLM-assisted migrations."""

from __future__ import annotations

from typing import Dict, Mapping, MutableMapping


# Conservative throughput estimate for an NVIDIA H100 when serving large
# instruction-tuned models.  The exact throughput depends on batching and model
# size; we use an easy-to-follow round number to keep the "receipt" readable.
H100_TOKENS_PER_SECOND = 1800

# Hypothetical on-demand hourly price (USD) for an H100 instance.  The actual
# cost fluctuates per provider; this value can be adjusted by callers via the
# ``hourly_rate`` argument.
H100_HOURLY_RATE = 4.10

# Default platform configuration: current infrastructure spend plus a
# 45 % markup to ensure migrations remain profitable by default.  Callers are
# free to override these values per request or via environment configuration.
DEFAULT_RESOURCE_COST = 10.58
DEFAULT_RESOURCE_CONTEXT = (
    "5h 43m left at current spend rate + tokens generation fee"
)
DEFAULT_MARKUP_RATE = 0.45


def _copy_usage(usage: Mapping[str, Mapping[str, int]]) -> Dict[str, Dict[str, int]]:
    copied: Dict[str, Dict[str, int]] = {}
    for key, value in usage.items():
        if key == "__totals__":
            continue
        copied[key] = dict(value)
    return copied


def estimate_h100_receipt(
    usage: Mapping[str, Mapping[str, int]],
    *,
    hourly_rate: float = H100_HOURLY_RATE,
    tokens_per_second: int = H100_TOKENS_PER_SECOND,
    resource_cost: float | int = 0.0,
    markup_rate: float = DEFAULT_MARKUP_RATE,
    resource_time_remaining: str | None = None,
    resource_notes: str | None = None,
) -> Dict[str, float | int | Dict[str, Dict[str, int]]]:
    """Build a cost breakdown using a simplified NVIDIA H100 model."""

    totals = usage.get("__totals__", {})
    prompt_tokens = int(totals.get("prompt_tokens", 0))
    completion_tokens = int(totals.get("completion_tokens", 0))
    total_tokens = prompt_tokens + completion_tokens

    if tokens_per_second <= 0:
        raise ValueError("tokens_per_second must be greater than zero")

    try:
        resource_cost_value = float(resource_cost)
    except (TypeError, ValueError) as exc:  # noqa: BLE001
        raise ValueError("resource_cost must be a numeric value") from exc

    markup = float(markup_rate)
    if markup < 0:
        raise ValueError("markup_rate must be greater than or equal to zero")

    estimated_seconds = total_tokens / float(tokens_per_second)
    estimated_hours = estimated_seconds / 3600.0
    token_generation_cost = estimated_hours * hourly_rate
    subtotal_cost = token_generation_cost + resource_cost_value
    margin_multiplier = 1.0 + markup
    suggested_price = subtotal_cost * margin_multiplier
    projected_profit = suggested_price - subtotal_cost

    receipt: MutableMapping[str, float | int | Dict[str, Dict[str, int]] | str] = {
        "hardware": "NVIDIA H100",
        "currency": "USD",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_seconds": estimated_seconds,
        "estimated_hours": estimated_hours,
        "estimated_cost": token_generation_cost,
        "token_generation_cost": token_generation_cost,
        "resource_cost": resource_cost_value,
        "subtotal_cost": subtotal_cost,
        "markup_rate": markup,
        "markup_percentage": markup * 100.0,
        "suggested_price": suggested_price,
        "projected_profit": projected_profit,
        "usage_breakdown": _copy_usage(usage),
        "hourly_rate": hourly_rate,
        "tokens_per_second": tokens_per_second,
        "margin_multiplier": margin_multiplier,
    }

    if resource_time_remaining is not None:
        receipt["resource_time_remaining"] = resource_time_remaining
    if resource_notes:
        receipt["resource_notes"] = resource_notes

    return dict(receipt)

