"""Background proactive reasoning loop.

Runs as a background anyio task that periodically polls registered trigger
handlers. When a trigger fires, it enqueues a proactive reasoning task.

Triggers are deduplicated by content hash to avoid spamming the user with
the same notification repeatedly.

Usage::

    loop = ProactiveLoop()

    @loop.trigger(interval_seconds=300)
    async def check_deadlines(ctx: ProactiveContext) -> ProactiveTrigger | None:
        overdue = await ctx.memory.retrieve("overdue tasks")
        if overdue:
            return ProactiveTrigger(
                prompt="You have overdue tasks. Remind the user.",
                priority=Priority.HIGH,
            )
        return None

    # Start the background loop (call inside an anyio task group)
    await loop.start()
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

import anyio
import anyio.abc

from nexagent.memory.tiered import TieredMemory

logger = logging.getLogger(__name__)


class Priority(int, Enum):
    LOW = 1
    MEDIUM = 5
    HIGH = 10
    CRITICAL = 100


@dataclass
class ProactiveTrigger:
    """A proactive reasoning trigger from a background handler."""

    prompt: str
    priority: Priority = Priority.MEDIUM
    ttl_seconds: float | None = 3600.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.prompt.encode()).hexdigest()[:16]


@dataclass
class ProactiveContext:
    """Context passed to trigger handlers."""

    memory: TieredMemory
    metadata: dict[str, Any] = field(default_factory=dict)


TriggerHandler = Callable[[ProactiveContext], Awaitable[ProactiveTrigger | None]]


@dataclass
class RegisteredTrigger:
    handler: TriggerHandler
    interval_seconds: float
    name: str
    last_run: float = 0.0
    enabled: bool = True


class ProactiveQueue:
    """Priority queue for proactive triggers.

    Deduplication is by content hash — the same trigger won't be enqueued
    more than once while it's still in the queue.
    """

    def __init__(self) -> None:
        self._items: list[ProactiveTrigger] = []
        self._seen_hashes: dict[str, float] = {}  # hash -> enqueue time

    def enqueue(self, trigger: ProactiveTrigger) -> bool:
        """Enqueue a trigger. Returns False if deduplicated."""
        now = time.monotonic()
        h = trigger.content_hash

        # Clean up expired seen hashes
        if trigger.ttl_seconds:
            self._seen_hashes = {
                k: v for k, v in self._seen_hashes.items()
                if now - v < (trigger.ttl_seconds or 3600)
            }

        if h in self._seen_hashes:
            return False

        self._seen_hashes[h] = now
        self._items.append(trigger)
        self._items.sort(key=lambda t: -t.priority)
        return True

    def dequeue(self) -> ProactiveTrigger | None:
        if not self._items:
            return None
        return self._items.pop(0)

    def __len__(self) -> int:
        return len(self._items)


class ProactiveLoop:
    """Background loop that polls trigger handlers and enqueues results.

    Parameters
    ----------
    memory:
        Shared memory tier passed to all trigger handlers.
    poll_interval:
        How often to check all triggers (seconds). Individual triggers may
        fire less frequently based on their own interval_seconds.
    paused:
        If True, the loop runs but skips execution (useful while agent is busy).
    """

    def __init__(
        self,
        memory: TieredMemory | None = None,
        poll_interval: float = 10.0,
    ) -> None:
        self._memory = memory or TieredMemory()
        self._poll_interval = poll_interval
        self._triggers: list[RegisteredTrigger] = []
        self._queue = ProactiveQueue()
        self._running = False
        self._paused = False
        self._task_scope: anyio.abc.CancelScope | None = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        handler: TriggerHandler,
        interval_seconds: float = 60.0,
        name: str | None = None,
    ) -> RegisteredTrigger:
        reg = RegisteredTrigger(
            handler=handler,
            interval_seconds=interval_seconds,
            name=name or handler.__name__,
        )
        self._triggers.append(reg)
        logger.debug("Registered proactive trigger: %s (interval=%ss)", reg.name, interval_seconds)
        return reg

    def trigger(
        self,
        interval_seconds: float = 60.0,
        name: str | None = None,
    ) -> Callable[[TriggerHandler], TriggerHandler]:
        """Decorator to register a trigger handler."""

        def decorator(fn: TriggerHandler) -> TriggerHandler:
            self.register(fn, interval_seconds=interval_seconds, name=name)
            return fn

        return decorator

    def unregister(self, name: str) -> bool:
        before = len(self._triggers)
        self._triggers = [t for t in self._triggers if t.name != name]
        return len(self._triggers) < before

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Run the proactive loop. Call this inside an anyio task group."""
        self._running = True
        logger.info("Proactive loop started (%d triggers)", len(self._triggers))
        try:
            while self._running:
                if not self._paused:
                    await self._tick()
                await anyio.sleep(self._poll_interval)
        except anyio.get_cancelled_exc_class():
            logger.info("Proactive loop cancelled")
        finally:
            self._running = False

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _tick(self) -> None:
        now = time.monotonic()
        ctx = ProactiveContext(memory=self._memory)

        for reg in self._triggers:
            if not reg.enabled:
                continue
            if now - reg.last_run < reg.interval_seconds:
                continue

            reg.last_run = now
            try:
                trigger = await reg.handler(ctx)
                if trigger is not None:
                    enqueued = self._queue.enqueue(trigger)
                    if enqueued:
                        logger.info(
                            "Proactive trigger enqueued: '%s' (priority=%s)",
                            trigger.prompt[:60],
                            trigger.priority.name,
                        )
            except Exception as exc:
                logger.error(
                    "Proactive trigger handler '%s' raised: %s", reg.name, exc, exc_info=True
                )

    def drain(self) -> list[ProactiveTrigger]:
        """Return and clear all pending triggers."""
        items = []
        while (t := self._queue.dequeue()) is not None:
            items.append(t)
        return items

    def peek(self) -> ProactiveTrigger | None:
        """Return the highest-priority pending trigger without removing it."""
        return self._queue._items[0] if self._queue._items else None

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def is_running(self) -> bool:
        return self._running
