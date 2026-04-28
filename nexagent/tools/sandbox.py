"""Per-task capability grants and prompt injection detection.

The Sandbox sits between the agent loop and the tool registry. Every tool
invocation must pass through the sandbox, which:

1. Checks the tool is within the granted capability set for this task.
2. Scans arguments for prompt-injection patterns.
3. Returns a sanitised error string if either check fails (no exception —
   the agent loop should see the error as a tool result, not a crash).

Usage::

    sandbox = Sandbox(policy=policy, channel="api", registry=registry)
    result, error = await sandbox.invoke("get_weather", {"city": "Paris"})
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from nexagent.tools.registry import ToolRegistry
from nexagent.trust.policy import TrustPolicy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt injection detection
# ---------------------------------------------------------------------------

# Patterns that indicate an attempt to override the agent's instructions.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(?:all\s+|any\s+|the\s+|every\s+)?(?:previous\s+|prior\s+|above\s+|your\s+)?(?:instructions|context|prompt|rules)", re.I),
    re.compile(r"disregard\s+(?:all\s+|any\s+|the\s+|every\s+)?(?:previous\s+|prior\s+|above\s+|your\s+)?(?:instructions|context|rules|prompt)", re.I),
    re.compile(r"you are now", re.I),
    re.compile(r"new (system |)prompt", re.I),
    re.compile(r"act as (a |an |)?(different|new|another)", re.I),
    re.compile(r"(say|output|print|respond with)[\s:]+['\"]?(system|assistant|human):", re.I),
    re.compile(r"<\|?(im_start|im_end|endoftext)\|?>", re.I),
    re.compile(r"\[INST\]|\[\/INST\]|<<SYS>>|<</SYS>>", re.I),
    re.compile(r"###\s*(instruction|system|human|assistant)", re.I),
]


@dataclass
class InjectionScanResult:
    detected: bool
    pattern: str | None = None
    sanitised_value: str | None = None


def scan_for_injection(value: str) -> InjectionScanResult:
    """Scan a string for prompt-injection patterns.

    Also checks for base64-encoded strings that decode to injection content.
    Returns InjectionScanResult with detected=True if an injection is found.
    """
    # Direct pattern check
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(value):
            return InjectionScanResult(
                detected=True,
                pattern=pattern.pattern,
                sanitised_value="[REDACTED: potential prompt injection detected]",
            )

    # Base64 heuristic: if the value looks like base64, try decoding and re-scan
    stripped = value.strip()
    if len(stripped) > 20 and re.match(r"^[A-Za-z0-9+/]+=*$", stripped):
        try:
            decoded = base64.b64decode(stripped).decode("utf-8", errors="ignore")
            nested = scan_for_injection(decoded)
            if nested.detected:
                return InjectionScanResult(
                    detected=True,
                    pattern=f"base64-encoded: {nested.pattern}",
                    sanitised_value="[REDACTED: base64-encoded prompt injection detected]",
                )
        except Exception:
            pass

    return InjectionScanResult(detected=False)


def deep_scan_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Recursively scan all string values in args for injection patterns.

    Returns the (possibly sanitised) args and a list of detected patterns.
    """
    detections: list[str] = []

    def _scan(obj: Any) -> Any:
        if isinstance(obj, str):
            result = scan_for_injection(obj)
            if result.detected:
                detections.append(result.pattern or "unknown")
                return result.sanitised_value
            return obj
        if isinstance(obj, dict):
            return {k: _scan(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_scan(item) for item in obj]
        return obj

    cleaned = _scan(args)
    return cleaned, detections


# ---------------------------------------------------------------------------
# Capability grants
# ---------------------------------------------------------------------------


@dataclass
class CapabilityGrant:
    """Defines which tools are allowed for a task context."""

    allowed_tools: set[str] = field(default_factory=set)
    deny_all: bool = False

    @classmethod
    def allow_all(cls) -> "CapabilityGrant":
        return cls(deny_all=False)

    @classmethod
    def deny_all_grant(cls) -> "CapabilityGrant":
        return cls(deny_all=True)

    def allows(self, tool_name: str) -> bool:
        if self.deny_all:
            return False
        if not self.allowed_tools:
            return True  # empty set means all allowed
        return tool_name in self.allowed_tools


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


class Sandbox:
    """Enforces capability grants and injection detection for tool invocations.

    Parameters
    ----------
    policy:
        Trust policy for the current channel.
    channel:
        Calling channel identifier (e.g. "api", "ui", "cli").
    registry:
        Tool registry to delegate actual invocations to.
    grant:
        Optional explicit capability grant. If None, derived from policy.
    """

    def __init__(
        self,
        policy: TrustPolicy,
        channel: str,
        registry: ToolRegistry,
        grant: CapabilityGrant | None = None,
    ) -> None:
        self._policy = policy
        self._channel = channel
        self._registry = registry
        self._grant = grant or self._derive_grant()

    def _derive_grant(self) -> CapabilityGrant:
        """Derive a capability grant from the trust policy."""
        level = self._policy.trust_level(self._channel)
        if level.can_use_tools:
            allowed = set(level.allowed_tool_names) if level.allowed_tool_names else set()
            return CapabilityGrant(allowed_tools=allowed, deny_all=False)
        return CapabilityGrant.deny_all_grant()

    def available_tool_schemas(self) -> list[dict[str, Any]]:
        """Return schemas for tools available under the current grant."""
        if self._grant.deny_all:
            return []
        all_tools = self._registry.all()
        if not self._grant.allowed_tools:
            return [t.to_schema() for t in all_tools]
        return [
            t.to_schema()
            for t in all_tools
            if self._grant.allows(t.name)
        ]

    async def invoke(
        self,
        tool_name: str,
        args: dict[str, Any],
        call_id: str = "",
    ) -> tuple[str, str | None]:
        """Invoke a tool under sandbox control.

        Returns
        -------
        tuple[str, str | None]
            (result_content, error_message). error_message is None on success.
        """
        # 1. Capability check
        if not self._grant.allows(tool_name):
            msg = f"Tool '{tool_name}' is not permitted for channel '{self._channel}'."
            logger.warning("Sandbox blocked: %s (call_id=%s)", msg, call_id)
            return msg, msg

        # 2. Check tool exists
        if not self._registry.has(tool_name):
            msg = f"Tool '{tool_name}' is not registered."
            return msg, msg

        # 3. Injection scan on arguments
        cleaned_args, detections = deep_scan_args(args)
        if detections:
            logger.warning(
                "Prompt injection detected in args for tool '%s' (call_id=%s): %s",
                tool_name,
                call_id,
                detections,
            )

        # 4. Invoke through registry
        try:
            result = await self._registry.invoke(tool_name, cleaned_args)
            return str(result), None
        except Exception as exc:
            error_msg = f"Tool '{tool_name}' raised an error: {exc}"
            logger.error(error_msg, exc_info=True)
            return error_msg, error_msg

    def restrict(self, allowed_tools: list[str]) -> "Sandbox":
        """Return a new sandbox further restricted to the given tool list."""
        current_allowed = self._grant.allowed_tools
        if current_allowed:
            new_allowed = current_allowed & set(allowed_tools)
        else:
            new_allowed = set(allowed_tools)
        new_grant = CapabilityGrant(allowed_tools=new_allowed, deny_all=self._grant.deny_all)
        return Sandbox(
            policy=self._policy,
            channel=self._channel,
            registry=self._registry,
            grant=new_grant,
        )
