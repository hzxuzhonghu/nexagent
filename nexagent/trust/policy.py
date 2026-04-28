"""Per-channel trust levels and autonomy dials.

Trust levels gate what the agent is allowed to do without human confirmation.
Each channel (SYSTEM, API, UI, CLI, PUBLIC) has a TrustLevel that specifies:
  - can_use_tools: whether any tool use is permitted
  - allowed_tool_names: whitelist (empty = all permitted tools)
  - max_tool_calls_per_turn: rate-limit on tool calls
  - requires_confirmation_for: set of tool tags needing explicit confirmation
  - can_exfiltrate: whether data may be sent to external services

Usage::

    policy = TrustPolicy.default()
    level = policy.trust_level("api")
    if not level.can_use_tools:
        raise PermissionError("This channel cannot use tools")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Channel(str, Enum):
    SYSTEM = "system"
    API = "api"
    UI = "ui"
    CLI = "cli"
    PUBLIC = "public"


@dataclass
class TrustLevel:
    """Capabilities and constraints for a given trust level."""

    name: str
    can_use_tools: bool
    can_exfiltrate: bool = False
    max_tool_calls_per_turn: int = 10
    allowed_tool_names: list[str] = field(default_factory=list)
    requires_confirmation_for: set[str] = field(default_factory=set)
    max_steps: int = 20
    metadata: dict[str, Any] = field(default_factory=dict)

    def allows_tool(self, tool_name: str, tool_tags: list[str] | None = None) -> bool:
        if not self.can_use_tools:
            return False
        if self.allowed_tool_names and tool_name not in self.allowed_tool_names:
            return False
        return True

    def needs_confirmation(self, tool_tags: list[str]) -> bool:
        return bool(self.requires_confirmation_for & set(tool_tags))


# Default trust level definitions
_DEFAULT_LEVELS: dict[str, TrustLevel] = {
    Channel.SYSTEM: TrustLevel(
        name="system",
        can_use_tools=True,
        can_exfiltrate=True,
        max_tool_calls_per_turn=100,
        allowed_tool_names=[],  # all tools
        requires_confirmation_for=set(),
        max_steps=50,
    ),
    Channel.API: TrustLevel(
        name="api",
        can_use_tools=True,
        can_exfiltrate=False,
        max_tool_calls_per_turn=20,
        allowed_tool_names=[],  # all registered tools
        requires_confirmation_for=set(),
        max_steps=20,
    ),
    Channel.UI: TrustLevel(
        name="ui",
        can_use_tools=True,
        can_exfiltrate=False,
        max_tool_calls_per_turn=10,
        allowed_tool_names=[],
        requires_confirmation_for={"destructive", "write", "delete"},
        max_steps=15,
    ),
    Channel.CLI: TrustLevel(
        name="cli",
        can_use_tools=True,
        can_exfiltrate=False,
        max_tool_calls_per_turn=5,
        allowed_tool_names=[],
        requires_confirmation_for={"destructive", "write", "delete", "external"},
        max_steps=10,
    ),
    Channel.PUBLIC: TrustLevel(
        name="public",
        can_use_tools=False,
        can_exfiltrate=False,
        max_tool_calls_per_turn=0,
        allowed_tool_names=[],
        requires_confirmation_for=set(),
        max_steps=5,
    ),
}


class TrustPolicy:
    """Manages trust levels across channels.

    Supports custom per-channel overrides for deployment-specific requirements.
    """

    def __init__(self, levels: dict[str, TrustLevel] | None = None) -> None:
        self._levels: dict[str, TrustLevel] = dict(_DEFAULT_LEVELS)
        if levels:
            self._levels.update(levels)

    @classmethod
    def default(cls) -> "TrustPolicy":
        return cls()

    @classmethod
    def strict(cls) -> "TrustPolicy":
        """Policy that requires confirmation for all tool calls on all channels."""
        strict_levels = {
            ch: TrustLevel(
                name=lvl.name,
                can_use_tools=lvl.can_use_tools,
                can_exfiltrate=False,
                max_tool_calls_per_turn=min(lvl.max_tool_calls_per_turn, 5),
                allowed_tool_names=lvl.allowed_tool_names,
                requires_confirmation_for={"read", "write", "delete", "destructive", "external"},
                max_steps=min(lvl.max_steps, 10),
            )
            for ch, lvl in _DEFAULT_LEVELS.items()
        }
        return cls(levels=strict_levels)

    def trust_level(self, channel: str) -> TrustLevel:
        """Return the TrustLevel for the given channel.

        Falls back to PUBLIC if the channel is unknown.
        """
        return self._levels.get(channel, _DEFAULT_LEVELS[Channel.PUBLIC])

    def override_channel(self, channel: str, level: TrustLevel) -> "TrustPolicy":
        """Return a new policy with an overridden level for one channel."""
        new_levels = dict(self._levels)
        new_levels[channel] = level
        return TrustPolicy(levels=new_levels)

    def escalate(self, channel: str, to_channel: str) -> "TrustLevel":
        """Return the trust level of to_channel applied in the context of channel.

        Used when a sub-operation temporarily requires higher trust (e.g. a
        SYSTEM-initiated sub-task runs in a USER context). This does NOT mutate
        the policy — it returns the target level directly.
        """
        return self.trust_level(to_channel)

    def describe(self) -> dict[str, Any]:
        return {
            ch: {
                "can_use_tools": lvl.can_use_tools,
                "can_exfiltrate": lvl.can_exfiltrate,
                "max_tool_calls_per_turn": lvl.max_tool_calls_per_turn,
                "max_steps": lvl.max_steps,
                "requires_confirmation_for": sorted(lvl.requires_confirmation_for),
            }
            for ch, lvl in self._levels.items()
        }
