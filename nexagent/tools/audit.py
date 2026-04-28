"""Audit log for all external tool calls.

Writes an append-only JSONL log of every tool invocation: caller, tool name,
sanitised arguments, result, timestamps, and cost metadata. This log is the
forensic trail for reviewing agent behaviour.

Usage::

    audit = AuditLog(path=Path("~/.nexagent/audit.jsonl"))
    await audit.record(
        session_id="abc123",
        tool_name="web_search",
        args={"query": "latest news"},
        result="...",
        call_id="call-1",
    )
    entries = audit.tail(20)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_AUDIT_PATH = Path.home() / ".nexagent" / "audit.jsonl"
MAX_ARG_VALUE_LEN = 1024  # truncate large argument values in the log
MAX_RESULT_LEN = 4096


@dataclass
class AuditEntry:
    """A single audit log entry."""

    call_id: str
    session_id: str
    tool_name: str
    args: dict[str, Any]
    result: str
    error: str | None
    timestamp_utc: str
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json_line(cls, line: str) -> "AuditEntry":
        return cls(**json.loads(line))


def _truncate_args(args: dict[str, Any]) -> dict[str, Any]:
    """Shallow truncation of long string values to keep the log readable."""
    out: dict[str, Any] = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > MAX_ARG_VALUE_LEN:
            out[k] = v[:MAX_ARG_VALUE_LEN] + "…[truncated]"
        else:
            out[k] = v
    return out


class AuditLog:
    """Append-only JSONL audit log.

    Writes are serialised through an asyncio.Lock to prevent interleaving
    in concurrent tool calls. The lock is per-instance; create one AuditLog
    per process.

    Parameters
    ----------
    path:
        Path to the JSONL file. Created on first write.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DEFAULT_AUDIT_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def record(
        self,
        session_id: str,
        tool_name: str,
        args: dict[str, Any],
        result: str,
        call_id: str = "",
        error: str | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Record a tool invocation. Thread-safe via asyncio.Lock."""
        entry = AuditEntry(
            call_id=call_id,
            session_id=session_id,
            tool_name=tool_name,
            args=_truncate_args(args),
            result=result[:MAX_RESULT_LEN] if len(result) > MAX_RESULT_LEN else result,
            error=error,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

        async with self._lock:
            try:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(entry.to_json_line() + "\n")
            except OSError as exc:
                logger.error("Failed to write audit log entry: %s", exc)

        return entry

    def tail(self, n: int = 50) -> list[AuditEntry]:
        """Return the last n entries from the log."""
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").splitlines()
        recent = lines[-n:]
        entries = []
        for line in recent:
            try:
                entries.append(AuditEntry.from_json_line(line))
            except Exception as exc:
                logger.warning("Corrupt audit log line: %s (%s)", line[:80], exc)
        return entries

    def search(
        self,
        session_id: str | None = None,
        tool_name: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Linear scan search by session_id and/or tool_name."""
        if not self._path.exists():
            return []

        results: list[AuditEntry] = []
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = AuditEntry.from_json_line(line)
                except Exception:
                    continue
                if session_id and entry.session_id != session_id:
                    continue
                if tool_name and entry.tool_name != tool_name:
                    continue
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

    def rotate(self, max_bytes: int = 50 * 1024 * 1024) -> bool:
        """Rotate the log if it exceeds max_bytes. Returns True if rotated."""
        if not self._path.exists():
            return False
        if os.path.getsize(self._path) < max_bytes:
            return False
        rotated = self._path.with_suffix(
            f".{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.jsonl"
        )
        self._path.rename(rotated)
        logger.info("Audit log rotated to %s", rotated)
        return True

    @property
    def path(self) -> Path:
        return self._path
