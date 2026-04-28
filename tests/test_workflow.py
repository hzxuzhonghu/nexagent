"""Tests for WorkflowContext, template interpolation, and Workflow DSL parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from nexagent.agents.registry import AgentConfig, AgentRegistry
from nexagent.agents.subagent import AgentOutput
from nexagent.agents.workflow import (
    WorkflowContext,
    WorkflowParser,
    interpolate_template,
)
from nexagent.tools.registry import ToolRegistry
from nexagent.trust.policy import TrustPolicy

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool_registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def policy() -> TrustPolicy:
    return TrustPolicy.default()


@pytest.fixture
def agent_registry(
    tool_registry: ToolRegistry, policy: TrustPolicy
) -> AgentRegistry:
    return AgentRegistry(tool_registry=tool_registry, policy=policy)


# ---------------------------------------------------------------------------
# WorkflowContext
# ---------------------------------------------------------------------------


class TestWorkflowContext:
    def test_set_and_get(self) -> None:
        ctx = WorkflowContext()
        ctx.set("topic", "climate")
        assert ctx.get("topic") == "climate"

    def test_get_with_default(self) -> None:
        ctx = WorkflowContext()
        assert ctx.get("missing", "fallback") == "fallback"

    def test_node_output_roundtrip(self) -> None:
        ctx = WorkflowContext()
        output = AgentOutput(content="data", agent="x", task_id="t1")
        ctx.set_node_output("node-a", output)
        assert ctx.get_node_output("node-a") is output

    def test_get_node_output_missing(self) -> None:
        ctx = WorkflowContext()
        assert ctx.get_node_output("nonexistent") is None

    def test_to_task_context_returns_dict(self) -> None:
        ctx = WorkflowContext()
        ctx.set("key", "value")
        tc = ctx.to_task_context()
        assert tc == {"key": "value"}

    def test_data_is_copy(self) -> None:
        ctx = WorkflowContext()
        ctx.set("a", 1)
        data = ctx.data
        data["a"] = 99
        assert ctx.get("a") == 1

    def test_initial_variables(self) -> None:
        ctx = WorkflowContext(variables={"seed": 42, "name": "test"})
        assert ctx.get("seed") == 42


# ---------------------------------------------------------------------------
# Template interpolation
# ---------------------------------------------------------------------------


class TestInterpolateTemplate:
    def test_simple_variable(self) -> None:
        ctx = WorkflowContext()
        ctx.set("topic", "AI")
        result = interpolate_template("Research {{topic}}", ctx)
        assert result == "Research AI"

    def test_node_output_content_reference(self) -> None:
        ctx = WorkflowContext()
        ctx.set_node_output("search", AgentOutput(content="found data", agent="s", task_id="t"))
        result = interpolate_template("Use: {{search.content}}", ctx)
        assert result == "Use: found data"

    def test_missing_variable_left_unrendered(self) -> None:
        ctx = WorkflowContext()
        result = interpolate_template("Hello {{name}}", ctx)
        assert result == "Hello {{name}}"

    def test_missing_node_output_left_unrendered(self) -> None:
        ctx = WorkflowContext()
        result = interpolate_template("Data: {{missing.content}}", ctx)
        assert result == "Data: {{missing.content}}"

    def test_multiple_variables(self) -> None:
        ctx = WorkflowContext()
        ctx.set("a", "1")
        ctx.set("b", "2")
        result = interpolate_template("{{a}} + {{b}}", ctx)
        assert result == "1 + 2"

    def test_no_template_returns_unchanged(self) -> None:
        ctx = WorkflowContext()
        result = interpolate_template("no templates here", ctx)
        assert result == "no templates here"

    def test_variable_with_numeric_value(self) -> None:
        ctx = WorkflowContext()
        ctx.set("count", 42)
        result = interpolate_template("Count: {{count}}", ctx)
        assert result == "Count: 42"


# ---------------------------------------------------------------------------
# Workflow DSL parser
# ---------------------------------------------------------------------------


class TestWorkflowParser:
    def test_parse_simple_workflow(self, agent_registry: AgentRegistry) -> None:
        agent_registry.register(AgentConfig(
            name="researcher",
            system_prompt="You research.",
        ))
        agent_registry.register(AgentConfig(
            name="writer",
            system_prompt="You write.",
        ))

        parser = WorkflowParser(agent_registry=agent_registry)
        graph, ctx = parser.parse_yaml("""
workflow: research_report
variables:
  topic: climate
nodes:
  - id: search
    agent: researcher
    prompt: "Find news about {{topic}}"
  - id: write
    agent: writer
    prompt: "Write about {{search.content}}"
    depends_on: [search]
""")
        assert len(graph.nodes) == 2
        # Variable interpolation should have been applied
        assert graph.nodes[0].task.prompt == "Find news about climate"
        # Node output references in later nodes are left unrendered at parse time
        # (since search hasn't run yet)
        assert "search.content" in graph.nodes[1].task.prompt

    def test_parse_file(self, agent_registry: AgentRegistry, tmp_path: Path) -> None:
        agent_registry.register(AgentConfig(
            name="agent",
            system_prompt="Be an agent.",
        ))
        yaml_path = tmp_path / "workflow.yaml"
        yaml_path.write_text("""
workflow: test
nodes:
  - id: single
    agent: agent
    prompt: "Do the thing"
""")
        parser = WorkflowParser(agent_registry=agent_registry)
        graph, ctx = parser.parse_file(yaml_path)
        assert len(graph.nodes) == 1
        assert graph.nodes[0].id == "single"

    def test_parse_with_max_retries(self, agent_registry: AgentRegistry) -> None:
        agent_registry.register(AgentConfig(name="risky", system_prompt="Try."))

        parser = WorkflowParser(agent_registry=agent_registry)
        graph, _ = parser.parse_yaml("""
workflow: risky
nodes:
  - id: faily
    agent: risky
    prompt: "Try hard"
    max_retries: 5
""")
        assert graph.nodes[0].max_retries == 5

    def test_parse_with_metadata(self, agent_registry: AgentRegistry) -> None:
        agent_registry.register(AgentConfig(name="m", system_prompt="M."))

        parser = WorkflowParser(agent_registry=agent_registry)
        graph, _ = parser.parse_yaml("""
workflow: meta
metadata:
  version: "1.0"
nodes:
  - id: n
    agent: m
    prompt: "hi"
    metadata:
      key: val
""")
        assert graph.metadata == {"version": "1.0"}

    def test_empty_yaml_raises(self, agent_registry: AgentRegistry) -> None:
        parser = WorkflowParser(agent_registry=agent_registry)
        with pytest.raises(ValueError, match="Empty"):
            parser.parse_yaml("")

    def test_parse_resolves_agent_names_to_configs(
        self, agent_registry: AgentRegistry
    ) -> None:
        agent_registry.register(AgentConfig(
            name="configured",
            system_prompt="Configured agent.",
            tools=["search"],
            max_steps=10,
        ))

        parser = WorkflowParser(agent_registry=agent_registry)
        graph, _ = parser.parse_yaml("""
workflow: test
nodes:
  - id: node1
    agent: configured
    prompt: "Do it"
""")
        node = graph.nodes[0]
        assert node.agent_config is not None
        assert node.agent_config.name == "configured"
        assert node.agent_config.max_steps == 10

    def test_parse_unknown_agent_raises_key_error(
        self, agent_registry: AgentRegistry
    ) -> None:
        parser = WorkflowParser(agent_registry=agent_registry)
        with pytest.raises(KeyError):
            parser.parse_yaml("""
workflow: test
nodes:
  - id: x
    agent: nonexistent
    prompt: "go"
""")
