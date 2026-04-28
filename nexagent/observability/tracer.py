"""OpenTelemetry-based tracer for NexAgent.

Provides a pre-configured tracer that wraps every reasoning step, tool call,
and inference routing decision in a span with semantic attributes.

Usage::

    tracer = get_tracer("nexagent.mymodule")
    with tracer.start_as_current_span("my.operation") as span:
        span.set_attribute("key", "value")
        ...
"""

from __future__ import annotations

import logging
import os
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

logger = logging.getLogger(__name__)

_PROVIDER_INITIALISED = False
_TRACER_PROVIDER: TracerProvider | None = None

OTEL_EXPORTER_OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
NEXAGENT_OTEL_CONSOLE = os.environ.get("NEXAGENT_OTEL_CONSOLE", "").lower() in ("1", "true")


def _init_provider() -> TracerProvider:
    global _PROVIDER_INITIALISED, _TRACER_PROVIDER

    if _PROVIDER_INITIALISED and _TRACER_PROVIDER is not None:
        return _TRACER_PROVIDER

    resource = Resource.create(
        {
            "service.name": "nexagent",
            "service.version": _get_version(),
        }
    )
    provider = TracerProvider(resource=resource)

    if OTEL_EXPORTER_OTLP_ENDPOINT:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=OTEL_EXPORTER_OTLP_ENDPOINT)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("OTEL OTLP exporter configured: %s", OTEL_EXPORTER_OTLP_ENDPOINT)
        except ImportError:
            logger.warning(
                "opentelemetry-exporter-otlp not installed; falling back to console exporter"
            )
            NEXAGENT_OTEL_CONSOLE_FLAG = True
        else:
            NEXAGENT_OTEL_CONSOLE_FLAG = NEXAGENT_OTEL_CONSOLE
    else:
        NEXAGENT_OTEL_CONSOLE_FLAG = NEXAGENT_OTEL_CONSOLE

    if NEXAGENT_OTEL_CONSOLE_FLAG:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _TRACER_PROVIDER = provider
    _PROVIDER_INITIALISED = True
    return provider


def _get_version() -> str:
    try:
        from importlib.metadata import version

        return version("nexagent")
    except Exception:
        return "0.0.0"


def get_tracer(name: str) -> trace.Tracer:
    """Return a configured OpenTelemetry tracer for the given instrument name."""
    _init_provider()
    return trace.get_tracer(name)


# ---------------------------------------------------------------------------
# Semantic attribute constants (follows OpenTelemetry semantic conventions)
# ---------------------------------------------------------------------------


class NexAgentAttributes:
    """Custom semantic attribute keys for NexAgent spans."""

    SESSION_ID = "nexagent.session_id"
    CHANNEL = "nexagent.channel"
    STEP = "nexagent.step"
    TOOL_NAME = "nexagent.tool.name"
    TOOL_CALL_ID = "nexagent.tool.call_id"
    INFERENCE_MODEL = "nexagent.inference.model"
    INFERENCE_ROUTED_TO = "nexagent.inference.routed_to"
    INFERENCE_CONFIDENCE = "nexagent.inference.confidence"
    MEMORY_TIER = "nexagent.memory.tier"
    MEMORY_QUERY = "nexagent.memory.query"
    MEMORY_HITS = "nexagent.memory.hits"
    COST_USD = "nexagent.cost.usd"
    COST_TOKENS_PROMPT = "nexagent.cost.tokens.prompt"
    COST_TOKENS_COMPLETION = "nexagent.cost.tokens.completion"
    AGENT_NAME = "nexagent.agent.name"
    COORDINATOR_RUN_ID = "nexagent.coordinator.run_id"
    COORDINATOR_NODE_ID = "nexagent.coordinator.node_id"


def span_from_dict(span: Any, attrs: dict[str, Any]) -> None:
    """Batch-apply a dict of attributes to a span."""
    for k, v in attrs.items():
        if v is not None:
            try:
                span.set_attribute(k, v)
            except Exception:
                pass
