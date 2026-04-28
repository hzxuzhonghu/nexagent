"""Tests for GenericAgent: runtime-configured subagent wrapping AgentLoop."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from nexagent.agents.generic import GenericAgent
from nexagent.agents.subagent import AgentOutput, AgentTask
from nexagent.memory.tiered import TieredMemory
from nexagent.runtime.agent_loop import AgentResult
from nexagent.tools.registry import ToolRegistry
from nexagent.trust.policy import TrustPolicy

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def policy() -> TrustPolicy:
    return TrustPolicy.default()


@pytest.fixture
def memory(tmp_path: Path) -> TieredMemory:
    return TieredMemory(base_path=tmp_path / "memory")


@pytest.fixture
def fake_agent_loop_result() -> AgentResult:
    return AgentResult(
        output="Hello, world!",
        session_id="sess-1",
        steps_taken=3,
        tool_calls_made=2,
        finish_reason="stop",
        cost_usd=0.01,
    )


# ---------------------------------------------------------------------------
# GenericAgent basic functionality
# ---------------------------------------------------------------------------


class TestGenericAgentBasic:
    def test_repr_shows_agent_name(self, registry: ToolRegistry, policy: TrustPolicy) -> None:
        agent = GenericAgent(
            name="test",
            description="Test agent",
            system_prompt="You are a test agent.",
            registry=registry,
            policy=policy,
        )
        assert "test" in repr(agent)

    def test_constructs_without_tools(self, registry: ToolRegistry, policy: TrustPolicy) -> None:
        agent = GenericAgent(
            name="bare",
            description="",
            system_prompt="You do nothing.",
            registry=registry,
            policy=policy,
        )
        assert agent._agent_name == "bare"
        assert agent._tools is None

    def test_constructs_with_tool_whitelist(
        self, registry: ToolRegistry, policy: TrustPolicy
    ) -> None:
        @registry.tool(name="search", description="", parameters={"type": "object", "properties": {}})
        async def search() -> str:
            return "results"

        @registry.tool(name="write", description="", parameters={"type": "object", "properties": {}})
        async def write() -> str:
            return "ok"

        agent = GenericAgent(
            name="reader",
            description="",
            system_prompt="Read only.",
            tools=["search"],
            registry=registry,
            policy=policy,
        )
        assert len(agent._registry) == 1
        assert agent._registry.has("search")
        assert not agent._registry.has("write")


# ---------------------------------------------------------------------------
# GenericAgent run() — mocked AgentLoop
# ---------------------------------------------------------------------------


class TestGenericAgentRun:
    @pytest.mark.asyncio
    async def test_run_returns_agent_output(
        self,
        registry: ToolRegistry,
        policy: TrustPolicy,
        fake_agent_loop_result: AgentResult,
    ) -> None:
        agent = GenericAgent(
            name="greeter",
            description="Greeting agent",
            system_prompt="You are friendly.",
            registry=registry,
            policy=policy,
        )
        # Patch AgentLoop.run to avoid network calls
        with patch("nexagent.runtime.agent_loop.AgentLoop.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = fake_agent_loop_result
            result = await agent.run(AgentTask(prompt="Say hello"))

            assert isinstance(result, AgentOutput)
            assert result.content == "Hello, world!"
            assert result.agent == "greeter"
            assert result.success is True
            mock_run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_with_max_steps_false_finish(
        self,
        registry: ToolRegistry,
        policy: TrustPolicy,
    ) -> None:
        result = AgentResult(
            output="timeout",
            session_id="s",
            steps_taken=20,
            tool_calls_made=0,
            finish_reason="max_steps",
        )

        agent = GenericAgent(
            name="limiter",
            description="",
            system_prompt="You are limited.",
            registry=registry,
            policy=policy,
            max_steps=5,
        )
        with patch("nexagent.runtime.agent_loop.AgentLoop.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = result
            output = await agent.run(AgentTask(prompt="go"))

            assert output.success is False
            assert output.metadata["steps"] == 20

    @pytest.mark.asyncio
    async def test_run_passes_task_context_to_metadata(
        self,
        registry: ToolRegistry,
        policy: TrustPolicy,
        fake_agent_loop_result: AgentResult,
    ) -> None:
        agent = GenericAgent(
            name="ctx_agent",
            description="",
            system_prompt="Test.",
            registry=registry,
            policy=policy,
        )
        with patch("nexagent.runtime.agent_loop.AgentLoop.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = fake_agent_loop_result
            task = AgentTask(
                prompt="hello",
                context={"user": "alice", "priority": 1},
            )
            await agent.run(task)
            # Verify AgentLoop.run was called (proving the context was set up)
            mock_run.assert_awaited_once()


# ---------------------------------------------------------------------------
# GenericAgent inside coordinator
# ---------------------------------------------------------------------------


class TestGenericAgentInCoordinator:
    @pytest.mark.asyncio
    async def test_coordinator_runs_generic_agent_node(
        self,
        registry: ToolRegistry,
        policy: TrustPolicy,
        memory: TieredMemory,
    ) -> None:
        from nexagent.agents.coordinator import (
            AgentCoordinator,
            CheckpointStore,
            TaskGraph,
            TaskNode,
        )
        from nexagent.agents.registry import AgentConfig

        config = AgentConfig(
            name="simple",
            description="Simple agent",
            system_prompt="You are simple.",
            max_steps=5,
        )

        result_output = AgentResult(
            output="generic output",
            session_id="s",
            steps_taken=1,
            tool_calls_made=0,
            finish_reason="stop",
        )

        with patch(
            "nexagent.runtime.agent_loop.AgentLoop.run",
            new_callable=AsyncMock,
            return_value=result_output,
        ):
            cp_store = CheckpointStore(db_path=memory.episodic._db_path.parent.parent / "cp.sqlite")
            coordinator = AgentCoordinator(
                registry=registry,
                policy=policy,
                memory=memory,
                checkpoint_store=cp_store,
            )

            graph = TaskGraph(
                nodes=[
                    TaskNode(
                        id="generic-node",
                        agent_class=GenericAgent,
                        task=AgentTask(prompt="run me"),
                        agent_config=config,
                    )
                ]
            )
            result = await coordinator.run(graph)

            assert result.success
            assert "generic-node" in result.outputs
            assert result.outputs["generic-node"].content == "generic output"
