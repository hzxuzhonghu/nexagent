"""Communication patterns for multi-agent coordination.

Provides builders and agent types for common coordination patterns:
- **FanOut**: parallel workers with optional collector
- **Supervisor**: dynamically extends the task graph at runtime
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from nexagent.agents.coordinator import TaskNode
from nexagent.agents.generic import GenericAgent
from nexagent.agents.registry import AgentRegistry
from nexagent.agents.subagent import AgentOutput, AgentTask, SubAgent
from nexagent.inference.router import InferenceRouter
from nexagent.memory.tiered import TieredMemory
from nexagent.tools.registry import ToolRegistry
from nexagent.trust.policy import TrustPolicy

logger = logging.getLogger(__name__)

# Type for the callback that supervisors use to inject new nodes.
AddNodesFn = Callable[[list[TaskNode]], Awaitable[None]]


# ---------------------------------------------------------------------------
# FanOut pattern
# ---------------------------------------------------------------------------


@dataclass
class FanOutConfig:
    """Configuration for a fan-out workflow."""

    worker_agent: str
    """Name of the registered agent to use for workers."""

    worker_prompts: list[str]
    """Prompt for each worker instance."""

    collector_agent: str | None = None
    """Optional agent name for a collector that merges worker outputs."""

    collector_prompt: str | None = None
    """Prompt for the collector. Uses ``{{worker_i.output}}`` references."""

    id_prefix: str = "worker"
    """Prefix for worker node IDs."""


class FanOutBuilder:
    """Builds TaskNodes for a fan-out pattern.

    Creates N parallel worker nodes (no inter-dependencies) plus an optional
    collector node that depends on all workers.
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        config: FanOutConfig,
    ) -> None:
        self._registry = agent_registry
        self._config = config

    def build_nodes(self) -> list[TaskNode]:
        """Build the list of TaskNodes for the fan-out pattern."""
        worker_config = self._registry.get(self._config.worker_agent)
        nodes: list[TaskNode] = []

        # Worker nodes — all independent
        worker_ids: list[str] = []
        for i, prompt in enumerate(self._config.worker_prompts):
            node_id = f"{self._config.id_prefix}_{i}"
            worker_ids.append(node_id)

            nodes.append(
                TaskNode(
                    id=node_id,
                    agent_class=GenericAgent,
                    task=AgentTask(prompt=prompt),
                    agent_config=worker_config,
                )
            )

        # Optional collector node
        if self._config.collector_agent and self._config.collector_prompt:
            collector_config = self._registry.get(self._config.collector_agent)
            nodes.append(
                TaskNode(
                    id="collector",
                    agent_class=GenericAgent,
                    task=AgentTask(prompt=self._config.collector_prompt),
                    depends_on=worker_ids,
                    agent_config=collector_config,
                )
            )

        return nodes


# ---------------------------------------------------------------------------
# Supervisor pattern
# ---------------------------------------------------------------------------


class SupervisorAgent(SubAgent):
    """A subagent that can dynamically extend the coordinator's task graph.

    The supervisor receives an ``add_nodes_fn`` callback via its task context
    that allows it to inject new nodes into the running coordinator graph.

    Usage::

        # In task.context:
        task.context = {
            "_add_nodes_fn": coordinator.add_nodes,
        }

        # Supervisor's run can then call:
        await add_nodes_fn([new_node1, new_node2])
    """

    name = "supervisor"
    description = "Supervises and dynamically extends the task graph"

    def __init__(
        self,
        registry: ToolRegistry,
        policy: TrustPolicy,
        memory: TieredMemory | None = None,
        router: InferenceRouter | None = None,
        channel: str = "api",
    ) -> None:
        super().__init__(
            registry=registry,
            policy=policy,
            memory=memory,
            router=router,
            channel=channel,
        )

    async def run(self, task: AgentTask) -> AgentOutput:
        """Execute the supervisor logic.

        The supervisor extracts ``_add_nodes_fn`` from the task context and
        uses its ``think()`` method to reason about sub-tasks, then
        dynamically injects new nodes as needed.
        """
        add_nodes_fn: AddNodesFn | None = task.context.get("_add_nodes_fn")
        worker_agent_name: str | None = task.context.get("_worker_agent_name")

        if not add_nodes_fn:
            return self._collect_output(
                content="No add_nodes_fn provided in task context.",
                task_id=task.task_id,
                success=False,
                error="Missing _add_nodes_fn in task.context",
            )

        # Use the prompt as a directive for what the supervisor should do
        reasoning = await self.think(
            prompt=task.prompt,
            system=(
                "You are a supervisor agent responsible for deciding how to "
                "break down tasks and delegate to workers. Respond with a "
                "clear plan of what sub-tasks to create."
            ),
        )

        # For now, the supervisor returns its reasoning as output.
        # In a more advanced implementation, it would parse the reasoning
        # to generate concrete sub-tasks and call add_nodes_fn.
        content = f"Supervisor plan:\n{reasoning}"
        if worker_agent_name:
            content += f"\n\nWorker agent available: {worker_agent_name}"

        return self._collect_output(
            content=content,
            task_id=task.task_id,
            success=True,
        )
