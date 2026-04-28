"""Tiered inference: local intent classifier + frontier model routing.

The router applies a two-stage decision:

  Stage 1 — Local classifier:
    Regex + heuristic keyword scoring. If confidence >= threshold, handle
    locally without an API call.

  Stage 2 — Frontier model:
    OpenAI-compatible POST /chat/completions via httpx. Used for complex
    reasoning, tool-calling, and low-confidence requests.

Usage::

    router = InferenceRouter()
    response = await router.route(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        tools=[],
        session_id="abc",
    )
    print(response.content)  # "4"
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

NEXAGENT_MODEL = os.environ.get("NEXAGENT_MODEL", "gpt-4o")
NEXAGENT_API_BASE = os.environ.get("NEXAGENT_API_BASE", "https://api.openai.com/v1")
NEXAGENT_API_KEY = os.environ.get("NEXAGENT_API_KEY", "")
NEXAGENT_LOCAL_THRESHOLD = float(os.environ.get("NEXAGENT_LOCAL_THRESHOLD", "0.7"))
NEXAGENT_MAX_STEPS = int(os.environ.get("NEXAGENT_MAX_STEPS", "20"))


@dataclass
class UsageInfo:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "local"
    latency_ms: float = 0.0


@dataclass
class RoutedResponse:
    """Normalised response from either the local classifier or frontier model."""

    content: str | None
    finish_reason: str  # "stop", "tool_calls", "length", "local"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: UsageInfo = field(default_factory=UsageInfo)
    routed_to: str = "frontier"

    @classmethod
    def local(cls, content: str) -> "RoutedResponse":
        return cls(
            content=content,
            finish_reason="stop",
            tool_calls=[],
            usage=UsageInfo(model="local"),
            routed_to="local",
        )


# ---------------------------------------------------------------------------
# Local intent classifier
# ---------------------------------------------------------------------------


@dataclass
class IntentPattern:
    pattern: re.Pattern[str]
    response_template: str
    confidence: float = 0.9


_LOCAL_PATTERNS: list[IntentPattern] = [
    IntentPattern(
        pattern=re.compile(r"\bwhat\s+(?:is|are)\s+(\d+)\s*[\+\-\*\/]\s*(\d+)\b", re.I),
        response_template="__ARITHMETIC__",
        confidence=0.95,
    ),
    IntentPattern(
        pattern=re.compile(r"\bwhat\s+(?:time|date|day)\s+(?:is\s+it|today)\b", re.I),
        response_template="__DATETIME__",
        confidence=0.9,
    ),
    IntentPattern(
        pattern=re.compile(r"\bhello|hi\b|^hey\b", re.I),
        response_template="Hello! How can I help you today?",
        confidence=0.85,
    ),
    IntentPattern(
        pattern=re.compile(r"\bthank(?:s| you)\b", re.I),
        response_template="You're welcome! Let me know if there's anything else I can help with.",
        confidence=0.9,
    ),
    IntentPattern(
        pattern=re.compile(r"\bwhat(?:'s|\s+is)\s+your\s+name\b", re.I),
        response_template="I'm NexAgent, your agentic personal assistant.",
        confidence=0.95,
    ),
    IntentPattern(
        pattern=re.compile(r"\bwhat\s+(?:can\s+you\s+do|are\s+you\s+capable\s+of)\b", re.I),
        response_template=(
            "I can help with research, task management, code review, scheduling, "
            "and much more. I use tools to interact with external services and "
            "remember context across our conversations."
        ),
        confidence=0.9,
    ),
    IntentPattern(
        pattern=re.compile(r"\bping\b", re.I),
        response_template="pong",
        confidence=0.99,
    ),
]


def _evaluate_arithmetic(expr: str) -> str:
    """Safely evaluate simple arithmetic expressions."""
    # Allow only digits, spaces, and arithmetic operators
    if re.fullmatch(r"[\d\s\+\-\*\/\(\)\.]+", expr):
        try:
            result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
            return str(result)
        except Exception:
            pass
    return "I couldn't compute that."


def _handle_arithmetic(text: str) -> str:
    match = re.search(r"(\d+\s*[\+\-\*\/]\s*\d+)", text)
    if match:
        return _evaluate_arithmetic(match.group(1))
    return "I couldn't parse that arithmetic expression."


def _handle_datetime() -> str:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return f"The current UTC time is {now.strftime('%Y-%m-%d %H:%M:%S UTC')}."


def classify_locally(text: str) -> tuple[str | None, float]:
    """Return (response, confidence) if the text matches a local pattern.

    Returns (None, 0.0) if no pattern matches confidently enough.
    """
    for intent in _LOCAL_PATTERNS:
        if intent.pattern.search(text):
            template = intent.response_template
            if template == "__ARITHMETIC__":
                return _handle_arithmetic(text), intent.confidence
            if template == "__DATETIME__":
                return _handle_datetime(), intent.confidence
            return template, intent.confidence
    return None, 0.0


# ---------------------------------------------------------------------------
# Frontier model client
# ---------------------------------------------------------------------------


async def call_frontier(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str = NEXAGENT_MODEL,
    api_base: str = NEXAGENT_API_BASE,
    api_key: str = NEXAGENT_API_KEY,
    timeout: float = 60.0,
) -> RoutedResponse:
    """Make a single call to the frontier model via OpenAI-compatible API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{api_base}/chat/completions",
            headers=headers,
            json=payload,
        )
    latency_ms = (time.perf_counter() - t0) * 1000

    if response.status_code != 200:
        logger.error(
            "Frontier model error %d: %s", response.status_code, response.text[:300]
        )
        return RoutedResponse(
            content=f"Inference error: HTTP {response.status_code}",
            finish_reason="error",
            usage=UsageInfo(model=model, latency_ms=latency_ms),
            routed_to="frontier",
        )

    data = response.json()
    choice = data["choices"][0]
    msg = choice.get("message", {})
    usage_raw = data.get("usage", {})

    return RoutedResponse(
        content=msg.get("content"),
        finish_reason=choice.get("finish_reason", "stop"),
        tool_calls=msg.get("tool_calls") or [],
        usage=UsageInfo(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            model=model,
            latency_ms=latency_ms,
        ),
        routed_to="frontier",
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class InferenceRouter:
    """Routes inference requests to local classifier or frontier model.

    Parameters
    ----------
    threshold:
        Minimum local classifier confidence to avoid a frontier call.
    model:
        Frontier model identifier.
    api_base:
        OpenAI-compatible API base URL.
    api_key:
        API key. Defaults to NEXAGENT_API_KEY env var.
    """

    def __init__(
        self,
        threshold: float = NEXAGENT_LOCAL_THRESHOLD,
        model: str = NEXAGENT_MODEL,
        api_base: str = NEXAGENT_API_BASE,
        api_key: str = NEXAGENT_API_KEY,
    ) -> None:
        self._threshold = threshold
        self._model = model
        self._api_base = api_base
        self._api_key = api_key
        self._local_calls = 0
        self._frontier_calls = 0

    async def route(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        session_id: str = "",
    ) -> RoutedResponse:
        """Route a request to local or frontier inference."""
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )

        # Only attempt local routing when there are no tools to call
        if not tools:
            local_response, confidence = classify_locally(last_user)
            if local_response is not None and confidence >= self._threshold:
                self._local_calls += 1
                logger.debug(
                    "Local routing (confidence=%.2f, session=%s)", confidence, session_id
                )
                return RoutedResponse.local(local_response)

        # Fall through to frontier
        self._frontier_calls += 1
        logger.debug("Frontier routing (session=%s)", session_id)
        return await call_frontier(
            messages=messages,
            tools=tools,
            model=self._model,
            api_base=self._api_base,
            api_key=self._api_key,
        )

    @property
    def stats(self) -> dict[str, int]:
        return {
            "local_calls": self._local_calls,
            "frontier_calls": self._frontier_calls,
        }
