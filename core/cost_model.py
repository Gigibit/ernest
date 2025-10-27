"""Utility helpers to estimate infrastructure costs for LLM-assisted migrations."""

from __future__ import annotations

from typing import Dict, Mapping


# Conservative throughput estimate for an NVIDIA H100 when serving large
# instruction-tuned models.  The exact throughput depends on batching and model
# size; we use an easy-to-follow round number to keep the "receipt" readable.
H100_TOKENS_PER_SECOND = 1800

# Hypothetical on-demand hourly price (USD) for an H100 instance.  The actual
# cost fluctuates per provider; this value can be adjusted by callers via the
# ``hourly_rate`` argument.
H100_HOURLY_RATE = 4.10


def estimate_h100_receipt(
    usage: Mapping[str, Mapping[str, int]],
    *,
    hourly_rate: float = H100_HOURLY_RATE,
    tokens_per_second: int = H100_TOKENS_PER_SECOND,
) -> Dict[str, float | int | Dict[str, Dict[str, int]]]:
    """Build a cost breakdown using a simplified NVIDIA H100 model."""

    totals = usage.get("__totals__", {})
    prompt_tokens = int(totals.get("prompt_tokens", 0))
    completion_tokens = int(totals.get("completion_tokens", 0))
    total_tokens = prompt_tokens + completion_tokens

    if tokens_per_second <= 0:
        raise ValueError("tokens_per_second must be greater than zero")

    estimated_seconds = total_tokens / float(tokens_per_second)
    estimated_hours = estimated_seconds / 3600.0
    estimated_cost = estimated_hours * hourly_rate

    return {
        "hardware": "NVIDIA H100",
        "currency": "USD",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_seconds": estimated_seconds,
        "estimated_hours": estimated_hours,
        "estimated_cost": estimated_cost,
        "usage_breakdown": {
            key: dict(value)
            for key, value in usage.items()
            if key != "__totals__"
        },
        "hourly_rate": hourly_rate,
        "tokens_per_second": tokens_per_second,
    }

