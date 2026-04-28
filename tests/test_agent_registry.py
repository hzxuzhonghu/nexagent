"""Tests for AgentConfig and AgentRegistry."""

from __future__ import annotations

from pathlib import Path

import pytest

from nexagent.agents.generic import GenericAgent
from nexagent.agents.registry import AgentConfig, AgentRegistry
from nexagent.memory.tiered import TieredMemory
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
def memory(tmp_path: Path) -> TieredMemory:
    return TieredMemory(base_path=tmp_path / "memory")


@pytest.fixture
def agent_registry(
    tool_registry: ToolRegistry, policy: TrustPolicy, memory: TieredMemory
) -> AgentRegistry:
    return AgentRegistry(
        tool_registry=tool_registry,
        policy=policy,
        memory=memory,
    )


# ---------------------------------------------------------------------------
# AgentConfig serialization and construction
# ---------------------------------------------------------------------------


class TestAgentConfig:
    def test_serialization(self) -> None:
        config = AgentConfig(
            name="researcher",
            description="Research assistant",
            system_prompt="You are a researcher.",
            tools=["search"],
            max_steps=10,
        )
        d = config.to_dict()
        assert d["name"] == "researcher"
        assert d["tools"] == ["search"]
        assert d["max_steps"] == 10

    def test_field_defaults(self) -> None:
        config = AgentConfig(name="minimal", system_prompt="Be minimal.")
        assert config.description == ""
        assert config.tools == []
        assert config.max_steps == 20
        assert config.model_override is None

    def test_from_dict(self) -> None:
        config = AgentConfig.from_dict(
            {"name": "writer", "system_prompt": "Write well.", "max_steps": 5}
        )
        assert config.name == "writer"
        assert config.max_steps == 5

    def test_with_model_override(self) -> None:
        config = AgentConfig(
            name="heavy",
            system_prompt="Think hard.",
            model_override="gpt-4o",
        )
        assert config.model_override == "gpt-4o"

    def test_instantiate_returns_generic_agent(
        self, tool_registry: ToolRegistry, policy: TrustPolicy
    ) -> None:
        config = AgentConfig(
            name="instantiated",
            description="Test",
            system_prompt="Be testy.",
            tools=["echo"],
        )
        agent = config.instantiate(registry=tool_registry, policy=policy)
        assert isinstance(agent, GenericAgent)
        assert agent._agent_name == "instantiated"


# ---------------------------------------------------------------------------
# AgentRegistry
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    def test_register_and_get(self, agent_registry: AgentRegistry) -> None:
        config = AgentConfig(name="test", system_prompt="Test prompt.")
        agent_registry.register(config)
        retrieved = agent_registry.get("test")
        assert retrieved is config

    def test_get_unknown_raises_key_error(self, agent_registry: AgentRegistry) -> None:
        with pytest.raises(KeyError, match="ghost"):
            agent_registry.get("ghost")

    def test_create_agent_returns_generic(
        self,
        agent_registry: AgentRegistry,
        tool_registry: ToolRegistry,
        policy: TrustPolicy,
    ) -> None:
        agent_registry.register(AgentConfig(name="creator", system_prompt="Create."))
        agent = agent_registry.create_agent("creator")
        assert isinstance(agent, GenericAgent)

    def test_names_property(self, agent_registry: AgentRegistry) -> None:
        agent_registry.register(AgentConfig(name="a", system_prompt="A."))
        agent_registry.register(AgentConfig(name="b", system_prompt="B."))
        assert set(agent_registry.names) == {"a", "b"}

    def test_len(self, agent_registry: AgentRegistry) -> None:
        assert len(agent_registry) == 0
        agent_registry.register(AgentConfig(name="x", system_prompt="X."))
        assert len(agent_registry) == 1

    def test_repr(self, agent_registry: AgentRegistry) -> None:
        agent_registry.register(AgentConfig(name="y", system_prompt="Y."))
        assert "y" in repr(agent_registry)

    def test_register_overwrites_with_warning(
        self, agent_registry: AgentRegistry, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging
        with caplog.at_level(logging.WARNING, logger="nexagent.agents.registry"):
            agent_registry.register(AgentConfig(name="dup", system_prompt="First."))
            agent_registry.register(AgentConfig(name="dup", system_prompt="Second."))
        assert any("dup" in r.message for r in caplog.records)
        # Second registration should overwrite
        assert agent_registry.get("dup").system_prompt == "Second."


# ---------------------------------------------------------------------------
# AgentRegistry.load_yaml
# ---------------------------------------------------------------------------


class TestAgentRegistryLoadYaml:
    def test_load_valid_yaml_file(self, agent_registry: AgentRegistry, tmp_path: Path) -> None:
        yaml_path = tmp_path / "agents.yaml"
        yaml_path.write_text("""
agents:
  - name: researcher
    description: "Research assistant"
    system_prompt: "You are a research expert."
    tools: [search, read_url]
    max_steps: 15
  - name: summarizer
    system_prompt: "Summarise the given text."
""")
        count = agent_registry.load_yaml(yaml_path)
        assert count == 2
        assert "researcher" in agent_registry.names
        assert "summarizer" in agent_registry.names
        config = agent_registry.get("researcher")
        assert config.max_steps == 15
        assert config.tools == ["search", "read_url"]

    def test_load_empty_file_returns_zero(self, agent_registry: AgentRegistry, tmp_path: Path) -> None:
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        count = agent_registry.load_yaml(empty)
        assert count == 0

    def test_load_missing_agents_key_returns_zero(
        self, agent_registry: AgentRegistry, tmp_path: Path
    ) -> None:
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("tools:\n  - echo\n")
        count = agent_registry.load_yaml(yaml_path)
        assert count == 0
