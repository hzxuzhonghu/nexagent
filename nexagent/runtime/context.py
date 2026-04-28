"""Session context and state management for the agent loop.

SessionContext is the single source of truth for a conversation. It is
immutable in the sense that mutation is always explicit (add_message,
set_metadata) and returns a new context or modifies state in a
documented way — never silently.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single message in the conversation history."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: Role
    content: str
    tool_call_id: str | None = None
    tool_name: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_api_dict(self) -> dict[str, Any]:
        """Serialise to the OpenAI-compatible chat message format."""
        d: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_name:
            d["name"] = self.tool_name
        return d


class SessionContext(BaseModel):
    """Complete state for a single agent session.

    Never store ``context.Context`` in a struct. This is our version of that
    rule for Python — SessionContext must flow through call signatures, not
    be captured in closures or class attributes of long-lived objects.
    """

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    channel: str = "api"
    messages: list[Message] = Field(default_factory=list)
    active_tool_names: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    step_count: int = 0

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def new(
        cls,
        channel: str = "api",
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "SessionContext":
        ctx = cls(channel=channel, metadata=metadata or {})
        if system_prompt:
            ctx.messages.append(Message(role=Role.SYSTEM, content=system_prompt))
        return ctx

    def add_user_message(self, content: str) -> "SessionContext":
        self.messages.append(Message(role=Role.USER, content=content))
        return self

    def add_assistant_message(self, content: str) -> "SessionContext":
        self.messages.append(Message(role=Role.ASSISTANT, content=content))
        return self

    def add_tool_result(
        self, tool_call_id: str, tool_name: str, content: str
    ) -> "SessionContext":
        self.messages.append(
            Message(
                role=Role.TOOL,
                content=content,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
            )
        )
        return self

    def api_messages(self) -> list[dict[str, Any]]:
        """Return the conversation formatted for an OpenAI-compatible API."""
        return [m.to_api_dict() for m in self.messages]

    def last_assistant_message(self) -> Message | None:
        for m in reversed(self.messages):
            if m.role == Role.ASSISTANT:
                return m
        return None

    def increment_step(self) -> int:
        self.step_count += 1
        return self.step_count

    def set_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def token_estimate(self) -> int:
        """Rough token count estimate (4 chars ≈ 1 token)."""
        total_chars = sum(len(m.content) for m in self.messages)
        return total_chars // 4
