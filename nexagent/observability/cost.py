"""Cost tracking per session.

Tracks prompt/completion tokens per model and maps to estimated USD cost.
Emits cost summaries and integrates with OpenTelemetry metrics.

Usage::

    tracker = CostTracker(session_id="abc123")
    tracker.record(UsageInfo(prompt_tokens=100, completion_tokens=50, model="gpt-4o"))
    print(tracker.total_usd())    # 0.000... USD
    print(tracker.summary())      # dict with per-model breakdown
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Price per 1M tokens in USD as of early 2025.
# Update this table as prices change — it's the only place to change.
_PRICE_TABLE: dict[str, dict[str, float]] = {
    "gpt-4o": {"prompt": 5.00, "completion": 15.00},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4-turbo": {"prompt": 10.00, "completion": 30.00},
    "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
    "claude-3-5-sonnet-20241022": {"prompt": 3.00, "completion": 15.00},
    "claude-3-haiku-20240307": {"prompt": 0.25, "completion": 1.25},
    "local": {"prompt": 0.0, "completion": 0.0},
}

_DEFAULT_PRICE = {"prompt": 10.0, "completion": 30.0}  # conservative fallback


@dataclass
class ModelUsage:
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost_usd(self) -> float:
        prices = _PRICE_TABLE.get(self.model, _DEFAULT_PRICE)
        prompt_cost = self.prompt_tokens * prices["prompt"] / 1_000_000
        completion_cost = self.completion_tokens * prices["completion"] / 1_000_000
        return prompt_cost + completion_cost


@dataclass
class UsageInfo:
    """Returned by InferenceRouter after each call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "local"
    latency_ms: float = 0.0


class CostTracker:
    """Accumulates token usage and cost for a single session.

    Parameters
    ----------
    session_id:
        Identifies the session for logging and export.
    budget_usd:
        Optional hard cap. When exceeded, budget_exceeded() returns True.
    """

    def __init__(
        self,
        session_id: str,
        budget_usd: float | None = None,
    ) -> None:
        self._session_id = session_id
        self._budget_usd = budget_usd
        self._usage: dict[str, ModelUsage] = {}
        self._total_latency_ms: float = 0.0
        self._calls: int = 0

    def record(self, usage: UsageInfo) -> None:
        """Record usage from a single inference call."""
        model = usage.model or "unknown"
        if model not in self._usage:
            self._usage[model] = ModelUsage(model=model)

        mu = self._usage[model]
        mu.prompt_tokens += usage.prompt_tokens
        mu.completion_tokens += usage.completion_tokens
        mu.calls += 1
        self._total_latency_ms += usage.latency_ms
        self._calls += 1

        cost = mu.cost_usd()
        logger.debug(
            "Cost recorded: model=%s prompt=%d completion=%d cumulative_cost=$%.6f",
            model,
            usage.prompt_tokens,
            usage.completion_tokens,
            self.total_usd(),
        )

        if self._budget_usd is not None and self.total_usd() > self._budget_usd:
            logger.warning(
                "Session %s exceeded budget: $%.4f > $%.4f",
                self._session_id,
                self.total_usd(),
                self._budget_usd,
            )

    def total_usd(self) -> float:
        return sum(mu.cost_usd() for mu in self._usage.values())

    def total_tokens(self) -> int:
        return sum(mu.total_tokens for mu in self._usage.values())

    def budget_exceeded(self) -> bool:
        if self._budget_usd is None:
            return False
        return self.total_usd() > self._budget_usd

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self._session_id,
            "total_calls": self._calls,
            "total_tokens": self.total_tokens(),
            "total_usd": round(self.total_usd(), 6),
            "total_latency_ms": round(self._total_latency_ms, 1),
            "budget_usd": self._budget_usd,
            "budget_exceeded": self.budget_exceeded(),
            "by_model": {
                model: {
                    "prompt_tokens": mu.prompt_tokens,
                    "completion_tokens": mu.completion_tokens,
                    "calls": mu.calls,
                    "cost_usd": round(mu.cost_usd(), 6),
                }
                for model, mu in self._usage.items()
            },
        }

    def reset(self) -> None:
        self._usage.clear()
        self._total_latency_ms = 0.0
        self._calls = 0


def price_for_model(model: str) -> dict[str, float]:
    """Return the price table entry for a model (prompt/completion per 1M tokens)."""
    return _PRICE_TABLE.get(model, _DEFAULT_PRICE)


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Quick-estimate USD cost without a tracker instance."""
    prices = price_for_model(model)
    return (
        prompt_tokens * prices["prompt"] / 1_000_000
        + completion_tokens * prices["completion"] / 1_000_000
    )
