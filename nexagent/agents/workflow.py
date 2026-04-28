"""Workflow context, template interpolation, and YAML DSL parser.

WorkflowContext provides structured data passing between coordinator nodes,
replacing the previous pattern of passing only raw string outputs. The DSL
parser converts YAML workflow definitions into executable TaskGraphs.

Usage::

    ctx = WorkflowContext()
    ctx.set("topic", "climate change")
    ctx.set_node_output("search", AgentOutput(content="..."))
    prompt = interpolate_template("Research {{topic}}", ctx)

    parser = WorkflowParser(agent_registry=agent_registry)
    graph, ctx = parser.parse_yaml(yaml_text)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from nexagent.agents.coordinator import GenericAgent, TaskGraph, TaskNode
from nexagent.agents.registry import AgentRegistry
from nexagent.agents.subagent import AgentTask

logger = logging.getLogger(__name__)

# Matches {{variable}} or {{node_id.field}}
_TEMPLATE_RE = re.compile(r"\{\{([\w.]+)\}\}")


# ---------------------------------------------------------------------------
# WorkflowContext
# ---------------------------------------------------------------------------


class WorkflowContext:
    """Shared data context across workflow nodes.

    Nodes can read/write arbitrary key-value pairs. The context also tracks
    node outputs so that downstream nodes can reference upstream results
    via template interpolation.
    """

    def __init__(self, variables: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = dict(variables or {})
        self._node_outputs: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set_node_output(self, node_id: str, output: Any) -> None:
        self._node_outputs[node_id] = output

    def get_node_output(self, node_id: str) -> Any | None:
        return self._node_outputs.get(node_id)

    @property
    def data(self) -> dict[str, Any]:
        return dict(self._data)

    def to_task_context(self) -> dict[str, Any]:
        """Return a dict suitable for passing to AgentTask.context."""
        return dict(self._data)


# ---------------------------------------------------------------------------
# Template interpolation
# ---------------------------------------------------------------------------


def interpolate_template(template: str, context: WorkflowContext) -> str:
    """Replace ``{{var}}`` and ``{{node_id.field}}`` with context values.

    Supports:
    - ``{{var}}`` → ``context.get("var")``
    - ``{{node_id.content}}`` → ``context.get_node_output("node_id").content``
    - ``{{node_id.field}}`` → resolves via getattr on AgentOutput or dict key lookup
    """

    def _replace(match: re.Match[str]) -> str:
        ref = match.group(1)

        # Check if it's a node output reference: node_id.field
        parts = ref.split(".", 1)
        if len(parts) == 2:
            node_id, attr = parts
            output = context.get_node_output(node_id)
            if output is None:
                return match.group(0)  # leave unrendered

            # AgentOutput is a dataclass — use getattr
            if hasattr(output, attr):
                val = getattr(output, attr)
            elif isinstance(output, dict) and attr in output:
                val = output[attr]
            else:
                return match.group(0)
            return str(val)

        # Simple variable
        val = context.get(ref)
        if val is None:
            return match.group(0)  # leave unrendered
        return str(val)

    return _TEMPLATE_RE.sub(_replace, template)


# ---------------------------------------------------------------------------
# Workflow DSL
# ---------------------------------------------------------------------------


@dataclass
class WorkflowNodeSpec:
    """A single node in a workflow specification."""

    id: str
    agent: str
    prompt: str
    depends_on: list[str] = field(default_factory=list)
    max_retries: int = 2
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowSpec:
    """Complete workflow specification parsed from YAML."""

    name: str
    nodes: list[WorkflowNodeSpec]
    variables: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class WorkflowParser:
    """Parses YAML workflow definitions into executable TaskGraphs.

    Parameters
    ----------
    agent_registry:
        Used to resolve agent names to AgentConfig instances.
    """

    def __init__(self, agent_registry: AgentRegistry) -> None:
        self._agent_registry = agent_registry

    def parse(self, spec: WorkflowSpec) -> tuple[TaskGraph, WorkflowContext]:
        """Parse a WorkflowSpec into a TaskGraph and initialized WorkflowContext."""
        context = WorkflowContext(variables=spec.variables)
        nodes: list[TaskNode] = []

        for node_spec in spec.nodes:
            config = self._agent_registry.get(node_spec.agent)
            prompt = interpolate_template(node_spec.prompt, context)

            task_node = TaskNode(
                id=node_spec.id,
                agent_class=GenericAgent,
                task=AgentTask(prompt=prompt, context=context.to_task_context()),
                depends_on=node_spec.depends_on,
                max_retries=node_spec.max_retries,
                agent_config=config,
            )
            nodes.append(task_node)

        graph = TaskGraph(nodes=nodes, metadata=spec.metadata)
        return graph, context

    def parse_yaml(self, yaml_text: str) -> tuple[TaskGraph, WorkflowContext]:
        """Parse a YAML string into a TaskGraph and WorkflowContext."""
        data = yaml.safe_load(yaml_text)
        return self._parse_dict(data)

    def parse_file(self, path: Path) -> tuple[TaskGraph, WorkflowContext]:
        """Parse a YAML file into a TaskGraph and WorkflowContext."""
        data = yaml.safe_load(Path(path).read_text())
        return self._parse_dict(data)

    def _parse_dict(self, data: dict[str, Any]) -> tuple[TaskGraph, WorkflowContext]:
        if not data:
            raise ValueError("Empty workflow definition")

        name = data.get("workflow", data.get("name", "unnamed"))
        variables = data.get("variables", {})
        metadata = data.get("metadata", {})

        raw_nodes = data.get("nodes", [])
        nodes: list[WorkflowNodeSpec] = []
        for item in raw_nodes:
            nodes.append(
                WorkflowNodeSpec(
                    id=item["id"],
                    agent=item["agent"],
                    prompt=item["prompt"],
                    depends_on=item.get("depends_on", []),
                    max_retries=item.get("max_retries", 2),
                    metadata=item.get("metadata", {}),
                )
            )

        spec = WorkflowSpec(
            name=name, nodes=nodes, variables=variables, metadata=metadata
        )
        return self.parse(spec)
