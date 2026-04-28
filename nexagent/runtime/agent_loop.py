"""Core async agent loop with tool-calling.

The loop follows the ReAct pattern:
  1. Receive a user prompt
  2. Call the inference router → get a response or tool-call request
  3. If tool call → dispatch via sandbox-protected registry → append result
  4. Repeat until the model emits a finish action or step limit is reached

Usage::

    loop = AgentLoop(context=ctx, policy=policy)
    result = await loop.run("What's on my schedule today?")
    print(result.output)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import anyio

from nexagent.inference.router import InferenceRouter, RoutedResponse
from nexagent.observability.cost import CostTracker
from nexagent.observability.tracer import get_tracer
from nexagent.runtime.context import Role, SessionContext
from nexagent.tools.audit import AuditLog
from nexagent.tools.registry import ToolRegistry
from nexagent.tools.sandbox import Sandbox
from nexagent.trust.policy import TrustPolicy

logger = logging.getLogger(__name__)

MAX_STEPS_DEFAULT = 20


@dataclass
class AgentResult:
    """Terminal result from an agent loop run."""

    output: str
    session_id: str
    steps_taken: int
    tool_calls_made: int
    finish_reason: str
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentLoop:
    """Async tool-calling agent loop.

    Parameters
    ----------
    context:
        Session context to operate on. The loop mutates this in place.
    policy:
        Trust policy governing which tools can be invoked.
    registry:
        Tool registry. If None, a default empty registry is created.
    router:
        Inference router. If None, a default router is created.
    max_steps:
        Hard limit on loop iterations to prevent infinite loops.
    """

    def __init__(
        self,
        context: SessionContext,
        policy: TrustPolicy,
        registry: ToolRegistry | None = None,
        router: InferenceRouter | None = None,
        max_steps: int = MAX_STEPS_DEFAULT,
        audit_log: AuditLog | None = None,
    ) -> None:
        self._ctx = context
        self._policy = policy
        self._registry = registry or ToolRegistry()
        self._router = router or InferenceRouter()
        self._max_steps = max_steps
        self._audit = audit_log or AuditLog()
        self._cost = CostTracker(session_id=context.session_id)
        self._tracer = get_tracer("nexagent.loop")
        self._tool_calls_made = 0

    async def run(self, prompt: str) -> AgentResult:
        """Execute the agent loop for a given user prompt."""
        with self._tracer.start_as_current_span("nexagent.loop.run") as span:
            span.set_attribute("session_id", self._ctx.session_id)
            span.set_attribute("channel", self._ctx.channel)

            self._ctx.add_user_message(prompt)
            sandbox = Sandbox(
                policy=self._policy,
                channel=self._ctx.channel,
                registry=self._registry,
            )

            finish_reason = "max_steps"
            final_output = ""

            for _ in range(self._max_steps):
                step = self._ctx.increment_step()
                logger.debug("Agent loop step %d | session=%s", step, self._ctx.session_id)

                with self._tracer.start_as_current_span("nexagent.loop.step") as step_span:
                    step_span.set_attribute("step", step)

                    response = await self._router.route(
                        messages=self._ctx.api_messages(),
                        tools=sandbox.available_tool_schemas(),
                        session_id=self._ctx.session_id,
                    )
                    self._cost.record(response.usage)

                    if response.finish_reason == "stop":
                        final_output = response.content or ""
                        self._ctx.add_assistant_message(final_output)
                        finish_reason = "stop"
                        break

                    if response.finish_reason == "tool_calls" and response.tool_calls:
                        self._ctx.add_assistant_message(response.content or "")
                        await self._dispatch_tool_calls(response.tool_calls, sandbox)
                    else:
                        # Unexpected finish reason — treat as terminal
                        final_output = response.content or ""
                        self._ctx.add_assistant_message(final_output)
                        finish_reason = response.finish_reason or "unknown"
                        break
            else:
                final_output = (
                    self._ctx.last_assistant_message()
                ).content if self._ctx.last_assistant_message() else ""

            span.set_attribute("steps_taken", self._ctx.step_count)
            span.set_attribute("tool_calls_made", self._tool_calls_made)
            span.set_attribute("finish_reason", finish_reason)

        return AgentResult(
            output=final_output,
            session_id=self._ctx.session_id,
            steps_taken=self._ctx.step_count,
            tool_calls_made=self._tool_calls_made,
            finish_reason=finish_reason,
            cost_usd=self._cost.total_usd(),
        )

    async def _dispatch_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        sandbox: Sandbox,
    ) -> None:
        """Execute tool calls concurrently under sandbox control."""
        async with anyio.create_task_group() as tg:
            results: list[tuple[str, str, str]] = []

            async def _invoke_one(tc: dict[str, Any]) -> None:
                call_id = tc.get("id", "unknown")
                fn = tc.get("function", {})
                name = fn.get("name", "")
                raw_args = fn.get("arguments", "{}")

                with self._tracer.start_as_current_span("nexagent.tool.invoke") as ts:
                    ts.set_attribute("tool.name", name)
                    ts.set_attribute("tool.call_id", call_id)

                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = {}

                    result_content, err = await sandbox.invoke(name, args, call_id=call_id)
                    if err:
                        ts.set_attribute("error", err)
                        logger.warning("Tool %s failed: %s", name, err)

                    await self._audit.record(
                        session_id=self._ctx.session_id,
                        tool_name=name,
                        args=args,
                        result=result_content,
                        call_id=call_id,
                        error=err,
                    )
                    self._tool_calls_made += 1
                    results.append((call_id, name, result_content))

            for tc in tool_calls:
                tg.start_soon(_invoke_one, tc)

        for call_id, name, content in results:
            self._ctx.add_tool_result(
                tool_call_id=call_id,
                tool_name=name,
                content=content,
            )

    @property
    def context(self) -> SessionContext:
        return self._ctx

    @property
    def cost_tracker(self) -> CostTracker:
        return self._cost


class AgentLoopBuilder:
    """Fluent builder for constructing an AgentLoop with custom components."""

    def __init__(self) -> None:
        self._context: SessionContext | None = None
        self._policy: TrustPolicy | None = None
        self._registry: ToolRegistry | None = None
        self._router: InferenceRouter | None = None
        self._max_steps: int = MAX_STEPS_DEFAULT
        self._audit: AuditLog | None = None

    def with_context(self, ctx: SessionContext) -> "AgentLoopBuilder":
        self._context = ctx
        return self

    def with_policy(self, policy: TrustPolicy) -> "AgentLoopBuilder":
        self._policy = policy
        return self

    def with_registry(self, registry: ToolRegistry) -> "AgentLoopBuilder":
        self._registry = registry
        return self

    def with_router(self, router: InferenceRouter) -> "AgentLoopBuilder":
        self._router = router
        return self

    def with_max_steps(self, n: int) -> "AgentLoopBuilder":
        self._max_steps = n
        return self

    def with_audit_log(self, audit: AuditLog) -> "AgentLoopBuilder":
        self._audit = audit
        return self

    def build(self) -> AgentLoop:
        if self._context is None:
            raise ValueError("SessionContext is required")
        if self._policy is None:
            from nexagent.trust.policy import TrustPolicy

            self._policy = TrustPolicy.default()
        return AgentLoop(
            context=self._context,
            policy=self._policy,
            registry=self._registry,
            router=self._router,
            max_steps=self._max_steps,
            audit_log=self._audit,
        )
