"""Agents package — coordinator, subagent base, and orchestration extensions."""

from nexagent.agents.coordinator import (
    AgentCoordinator,
    CoordinatorResult,
    NodeStatus,
    TaskGraph,
    TaskNode,
)
from nexagent.agents.generic import GenericAgent
from nexagent.agents.registry import AgentConfig, AgentRegistry
from nexagent.agents.subagent import AgentOutput, AgentTask, SubAgent
from nexagent.agents.workflow import (
    WorkflowContext,
    WorkflowNodeSpec,
    WorkflowParser,
    WorkflowSpec,
    interpolate_template,
)
from nexagent.agents.workspace import AgentPersona, AgentWorkspace

__all__ = [
    "AgentConfig",
    "AgentCoordinator",
    "AgentOutput",
    "AgentPersona",
    "AgentRegistry",
    "AgentTask",
    "AgentWorkspace",
    "CoordinatorResult",
    "GenericAgent",
    "NodeStatus",
    "SubAgent",
    "TaskGraph",
    "TaskNode",
    "WorkflowContext",
    "WorkflowNodeSpec",
    "WorkflowParser",
    "WorkflowSpec",
    "interpolate_template",
]
