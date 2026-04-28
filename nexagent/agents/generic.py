"""Generic subagent configured at runtime with a system prompt and tool set.

GenericAgent wraps the existing AgentLoop so that agents can be defined
entirely through configuration (name, system prompt, tool whitelist) without
writing Python subclasses.

Usage::

    agent = GenericAgent(
        name="researcher",
        description="Research assistant",
        system_prompt="You are a research expert...",
        tools=["web_search", "read_url"],
        registry=registry,
        policy=policy,
    )
    output = await agent.run(AgentTask(prompt="Find X"))
"""

from __future__ import annotations

import logging

from nexagent.agents.subagent import AgentOutput, AgentTask, SubAgent
from nexagent.agents.workspace import AgentWorkspace
from nexagent.inference.models import ModelPool
from nexagent.inference.router import InferenceRouter
from nexagent.memory.tiered import TieredMemory
from nexagent.runtime.context import SessionContext
from nexagent.tools.registry import ToolRegistry
from nexagent.trust.policy import TrustPolicy

logger = logging.getLogger(__name__)


class GenericAgent(SubAgent):
    """A subagent configured at runtime with a system prompt and tool whitelist.

    Rather than subclassing SubAgent and implementing ``run()``, a GenericAgent
    is fully specified by its constructor arguments. The ``run()`` method
    creates a SessionContext with the agent's system prompt, runs the ReAct
    AgentLoop, and converts the AgentResult to an AgentOutput.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        system_prompt: str,
        tools: list[str] | None = None,
        max_steps: int = 20,
        registry: ToolRegistry,
        policy: TrustPolicy,
        memory: TieredMemory | None = None,
        router: InferenceRouter | None = None,
        channel: str = "api",
        model_pool: ModelPool | None = None,
        model_id: str | None = None,
        workspace: AgentWorkspace | None = None,
    ) -> None:
        self._agent_name = name
        self._description = description
        self._tools = tools
        self._max_steps = max_steps
        self._channel = channel

        # Compose system prompt from workspace if available
        if workspace is not None:
            self._system_prompt = workspace.compose_system_prompt(system_prompt)
        else:
            self._system_prompt = system_prompt

        # If model_pool + model_id provided and no router, create one
        effective_router = router
        if effective_router is None and model_pool is not None and model_id is not None:
            effective_router = InferenceRouter(
                model_pool=model_pool,
                model_id=model_id,
            )

        # If a tool whitelist is specified, create a subsetted registry
        effective_registry: ToolRegistry = registry
        if tools:
            effective_registry = registry.subset(tools)

        super().__init__(
            registry=effective_registry,
            policy=policy,
            memory=memory,
            router=effective_router,
            channel=channel,
        )

    async def run(self, task: AgentTask) -> AgentOutput:
        """Execute the agent by running the ReAct loop."""
        from nexagent.runtime.agent_loop import AgentLoop

        ctx = SessionContext.new(
            channel=self._channel,
            system_prompt=self._system_prompt,
        )
        ctx.set_metadata("agent_name", self._agent_name)
        ctx.set_metadata("task_id", task.task_id)

        if task.context:
            for key, value in task.context.items():
                ctx.set_metadata(key, value)

        loop = AgentLoop(
            context=ctx,
            policy=self._policy,
            registry=self._registry,
            router=self._router,
            max_steps=self._max_steps,
        )

        result = await loop.run(task.prompt)

        return AgentOutput(
            content=result.output,
            agent=self._agent_name,
            task_id=task.task_id,
            tool_calls_made=result.tool_calls_made,
            success=result.finish_reason == "stop",
            metadata={
                "steps": result.steps_taken,
                "finish_reason": result.finish_reason,
                "cost_usd": result.cost_usd,
            },
        )

    def __repr__(self) -> str:
        return f"GenericAgent(name={self._agent_name!r})"
