"""Base subagent class for specialised agents.

Subagents are focused units of capability (e.g. ResearchAgent, CodeAgent).
They share a common interface for tool execution, memory access, and
structured output emission.

Usage::

    class ResearchAgent(SubAgent):
        name = "research"
        description = "Searches the web and summarises results."

        async def run(self, task: AgentTask) -> AgentOutput:
            results = await self.invoke_tool("web_search", {"query": task.prompt})
            summary = await self.think(f"Summarise: {results}")
            return AgentOutput(content=summary, agent=self.name)
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from nexagent.inference.router import InferenceRouter, RoutedResponse
from nexagent.memory.tiered import TieredMemory
from nexagent.tools.registry import ToolRegistry
from nexagent.tools.sandbox import Sandbox
from nexagent.trust.policy import TrustPolicy

logger = logging.getLogger(__name__)


@dataclass
class AgentTask:
    """Input to a subagent."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    parent_session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentOutput:
    """Structured output from a subagent."""

    content: str
    agent: str
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_calls_made: int = 0
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None


class SubAgent(ABC):
    """Abstract base class for all specialised subagents.

    Concrete subagents must implement:
    - ``name``: class-level string identifier
    - ``description``: human-readable description
    - ``run(task)``: the core async execution method

    The base class provides:
    - ``invoke_tool``: sandboxed tool invocation
    - ``think``: single-shot inference without tool calls
    - ``remember`` / ``recall``: memory tier access
    - ``emit_artifact``: structured output attachment
    """

    name: str = "base"
    description: str = "Base subagent — do not use directly"

    def __init__(
        self,
        registry: ToolRegistry,
        policy: TrustPolicy,
        memory: TieredMemory | None = None,
        router: InferenceRouter | None = None,
        channel: str = "api",
    ) -> None:
        self._registry = registry
        self._policy = policy
        self._memory = memory or TieredMemory()
        self._router = router or InferenceRouter()
        self._channel = channel
        self._sandbox = Sandbox(
            policy=policy,
            channel=channel,
            registry=registry,
        )
        self._artifacts: list[dict[str, Any]] = []
        self._tool_calls: int = 0

    @abstractmethod
    async def run(self, task: AgentTask) -> AgentOutput:
        """Execute the agent on the given task. Must be implemented."""

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def invoke_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        call_id: str | None = None,
    ) -> tuple[str, str | None]:
        """Invoke a tool under sandbox control.

        Returns (result_str, error_str). error_str is None on success.
        """
        cid = call_id or str(uuid.uuid4())
        result, error = await self._sandbox.invoke(tool_name, args, call_id=cid)
        self._tool_calls += 1
        logger.debug(
            "SubAgent[%s] tool=%s call_id=%s error=%s", self.name, tool_name, cid, error
        )
        return result, error

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    async def think(
        self,
        prompt: str,
        system: str | None = None,
        context_messages: list[dict[str, Any]] | None = None,
    ) -> str:
        """Single-shot inference with no tool calls.

        Returns the model's text response.
        """
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if context_messages:
            messages.extend(context_messages)
        messages.append({"role": "user", "content": prompt})

        response: RoutedResponse = await self._router.route(
            messages=messages,
            tools=[],
        )
        return response.content or ""

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def remember(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store a value in working memory."""
        self._memory.working.set(key, value, ttl=ttl)

    def recall(self, key: str) -> Any:
        """Retrieve a value from working memory by key."""
        return self._memory.working.get(key)

    async def retrieve_memory(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int = 5,
    ) -> list[Any]:
        """Retrieve relevant memories across all tiers."""
        items = await self._memory.retrieve(query, session_id=session_id, top_k=top_k)
        return [item.content for item in items]

    # ------------------------------------------------------------------
    # Artifact emission
    # ------------------------------------------------------------------

    def emit_artifact(
        self,
        artifact_type: str,
        content: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Attach a structured artifact to the current run."""
        self._artifacts.append(
            {
                "type": artifact_type,
                "content": content,
                "metadata": metadata or {},
            }
        )

    def _collect_output(
        self,
        content: str,
        task_id: str,
        success: bool = True,
        error: str | None = None,
    ) -> AgentOutput:
        output = AgentOutput(
            content=content,
            agent=self.name,
            task_id=task_id,
            tool_calls_made=self._tool_calls,
            artifacts=list(self._artifacts),
            success=success,
            error=error,
        )
        self._artifacts.clear()
        self._tool_calls = 0
        return output

    def __repr__(self) -> str:
        return f"SubAgent(name={self.name!r})"
