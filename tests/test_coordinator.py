"""Tests for AgentCoordinator: subagents, tasks, checkpoints, and failure handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nexagent.agents.coordinator import (
    AgentCoordinator,
    CheckpointStore,
    CoordinatorResult,
    NodeStatus,
    TaskGraph,
    TaskNode,
)
from nexagent.agents.subagent import AgentOutput, AgentTask, SubAgent
from nexagent.memory.tiered import TieredMemory
from nexagent.tools.registry import ToolRegistry
from nexagent.trust.policy import TrustPolicy


# ---------------------------------------------------------------------------
# Stub SubAgent implementations used across tests
# ---------------------------------------------------------------------------


class EchoAgent(SubAgent):
    """Returns the task prompt as content. Always succeeds."""

    name = "echo"
    description = "Echoes the task prompt."

    async def run(self, task: AgentTask) -> AgentOutput:
        return AgentOutput(
            content=f"echo:{task.prompt}",
            agent=self.name,
            task_id=task.task_id,
            success=True,
        )


class UpperCaseAgent(SubAgent):
    """Returns the uppercased prompt. A distinct agent type for multi-agent tests."""

    name = "uppercase"
    description = "Uppercases the task prompt."

    async def run(self, task: AgentTask) -> AgentOutput:
        return AgentOutput(
            content=task.prompt.upper(),
            agent=self.name,
            task_id=task.task_id,
            success=True,
        )


class FailingAgent(SubAgent):
    """Always returns a failure output (success=False)."""

    name = "failing"
    description = "Always fails."

    async def run(self, task: AgentTask) -> AgentOutput:
        return AgentOutput(
            content="",
            agent=self.name,
            task_id=task.task_id,
            success=False,
            error="intentional failure",
        )


class ExplodingAgent(SubAgent):
    """Always raises an exception. Used to test exception-based failure handling."""

    name = "exploding"
    description = "Always raises."

    async def run(self, task: AgentTask) -> AgentOutput:
        raise RuntimeError("This agent always explodes")


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
def checkpoint_store(tmp_path: Path) -> CheckpointStore:
    return CheckpointStore(db_path=tmp_path / "checkpoints.sqlite")


@pytest.fixture
def coordinator(
    registry: ToolRegistry,
    policy: TrustPolicy,
    memory: TieredMemory,
    checkpoint_store: CheckpointStore,
) -> AgentCoordinator:
    return AgentCoordinator(
        registry=registry,
        policy=policy,
        memory=memory,
        checkpoint_store=checkpoint_store,
    )


# ---------------------------------------------------------------------------
# TaskGraph unit tests (no I/O)
# ---------------------------------------------------------------------------


class TestTaskGraph:
    def test_node_map_keys_match_node_ids(self) -> None:
        graph = TaskGraph(
            nodes=[
                TaskNode(id="a", agent_class=EchoAgent, task=AgentTask(prompt="")),
                TaskNode(id="b", agent_class=EchoAgent, task=AgentTask(prompt="")),
            ]
        )
        node_map = graph.node_map()
        assert set(node_map.keys()) == {"a", "b"}

    def test_ready_nodes_returns_only_nodes_with_no_pending_deps(self) -> None:
        graph = TaskGraph(
            nodes=[
                TaskNode(id="x", agent_class=EchoAgent, task=AgentTask(prompt="")),
                TaskNode(
                    id="y",
                    agent_class=EchoAgent,
                    task=AgentTask(prompt=""),
                    depends_on=["x"],
                ),
            ]
        )
        ready = graph.ready_nodes()
        ready_ids = [n.id for n in ready]
        assert "x" in ready_ids
        assert "y" not in ready_ids

    def test_ready_nodes_unblocked_after_dependency_done(self) -> None:
        node_x = TaskNode(id="x", agent_class=EchoAgent, task=AgentTask(prompt=""))
        node_y = TaskNode(
            id="y", agent_class=EchoAgent, task=AgentTask(prompt=""), depends_on=["x"]
        )
        graph = TaskGraph(nodes=[node_x, node_y])
        node_x.status = NodeStatus.DONE
        ready = graph.ready_nodes()
        assert any(n.id == "y" for n in ready)

    def test_is_complete_true_when_all_nodes_terminal(self) -> None:
        graph = TaskGraph(
            nodes=[
                TaskNode(
                    id="n1",
                    agent_class=EchoAgent,
                    task=AgentTask(prompt=""),
                    status=NodeStatus.DONE,
                ),
                TaskNode(
                    id="n2",
                    agent_class=EchoAgent,
                    task=AgentTask(prompt=""),
                    status=NodeStatus.FAILED,
                ),
            ]
        )
        assert graph.is_complete() is True

    def test_is_complete_false_when_pending_node_exists(self) -> None:
        graph = TaskGraph(
            nodes=[
                TaskNode(
                    id="n1",
                    agent_class=EchoAgent,
                    task=AgentTask(prompt=""),
                    status=NodeStatus.DONE,
                ),
                TaskNode(
                    id="n2",
                    agent_class=EchoAgent,
                    task=AgentTask(prompt=""),
                    status=NodeStatus.PENDING,
                ),
            ]
        )
        assert graph.is_complete() is False

    def test_failed_nodes_returns_only_failed(self) -> None:
        graph = TaskGraph(
            nodes=[
                TaskNode(
                    id="ok",
                    agent_class=EchoAgent,
                    task=AgentTask(prompt=""),
                    status=NodeStatus.DONE,
                ),
                TaskNode(
                    id="bad",
                    agent_class=EchoAgent,
                    task=AgentTask(prompt=""),
                    status=NodeStatus.FAILED,
                ),
            ]
        )
        failed = graph.failed_nodes()
        assert len(failed) == 1
        assert failed[0].id == "bad"


# ---------------------------------------------------------------------------
# Coordinator — subagent registration via TaskGraph
# ---------------------------------------------------------------------------


class TestCoordinatorSubagents:
    @pytest.mark.asyncio
    async def test_coordinator_accepts_multiple_agent_types(
        self, coordinator: AgentCoordinator
    ) -> None:
        """TaskGraph can contain nodes with different agent_class types."""
        graph = TaskGraph(
            nodes=[
                TaskNode(id="echo-node", agent_class=EchoAgent, task=AgentTask(prompt="hello")),
                TaskNode(id="upper-node", agent_class=UpperCaseAgent, task=AgentTask(prompt="world")),
            ]
        )
        result = await coordinator.run(graph)
        assert result.success
        assert "echo-node" in result.outputs
        assert "upper-node" in result.outputs

    @pytest.mark.asyncio
    async def test_coordinator_runs_single_node_graph(
        self, coordinator: AgentCoordinator
    ) -> None:
        graph = TaskGraph(
            nodes=[TaskNode(id="solo", agent_class=EchoAgent, task=AgentTask(prompt="single"))]
        )
        result = await coordinator.run(graph)
        assert result.success
        assert len(result.outputs) == 1


# ---------------------------------------------------------------------------
# Coordinator — task assignment and result collection
# ---------------------------------------------------------------------------


class TestCoordinatorTaskResults:
    @pytest.mark.asyncio
    async def test_output_content_matches_agent_logic(
        self, coordinator: AgentCoordinator
    ) -> None:
        graph = TaskGraph(
            nodes=[
                TaskNode(
                    id="echo-task",
                    agent_class=EchoAgent,
                    task=AgentTask(prompt="test prompt"),
                )
            ]
        )
        result = await coordinator.run(graph)
        assert result.outputs["echo-task"].content == "echo:test prompt"

    @pytest.mark.asyncio
    async def test_results_keyed_by_node_id(self, coordinator: AgentCoordinator) -> None:
        graph = TaskGraph(
            nodes=[
                TaskNode(id="node-alpha", agent_class=EchoAgent, task=AgentTask(prompt="a")),
                TaskNode(id="node-beta", agent_class=UpperCaseAgent, task=AgentTask(prompt="b")),
            ]
        )
        result = await coordinator.run(graph)
        assert set(result.outputs.keys()) == {"node-alpha", "node-beta"}
        assert result.outputs["node-beta"].content == "B"

    @pytest.mark.asyncio
    async def test_dependent_node_runs_after_dependency(
        self, coordinator: AgentCoordinator
    ) -> None:
        """Node 'second' must only run after 'first' has completed."""
        execution_order: list[str] = []

        class OrderTrackingFirst(SubAgent):
            name = "first"
            description = ""

            async def run(self, task: AgentTask) -> AgentOutput:
                execution_order.append("first")
                return AgentOutput(content="1", agent=self.name, task_id=task.task_id)

        class OrderTrackingSecond(SubAgent):
            name = "second"
            description = ""

            async def run(self, task: AgentTask) -> AgentOutput:
                execution_order.append("second")
                return AgentOutput(content="2", agent=self.name, task_id=task.task_id)

        graph = TaskGraph(
            nodes=[
                TaskNode(id="first", agent_class=OrderTrackingFirst, task=AgentTask(prompt="")),
                TaskNode(
                    id="second",
                    agent_class=OrderTrackingSecond,
                    task=AgentTask(prompt=""),
                    depends_on=["first"],
                ),
            ]
        )
        result = await coordinator.run(graph)
        assert result.success
        assert execution_order.index("first") < execution_order.index("second")

    @pytest.mark.asyncio
    async def test_submit_and_wait_is_equivalent_to_run(
        self, coordinator: AgentCoordinator
    ) -> None:
        graph = TaskGraph(
            nodes=[TaskNode(id="n", agent_class=EchoAgent, task=AgentTask(prompt="hi"))]
        )
        result = await coordinator.submit_and_wait(graph)
        assert result.success
        assert result.outputs["n"].content == "echo:hi"


# ---------------------------------------------------------------------------
# CheckpointStore — save and load
# ---------------------------------------------------------------------------


class TestCheckpointStore:
    def test_save_and_load_completed_node(self, tmp_path: Path) -> None:
        store = CheckpointStore(db_path=tmp_path / "cp.sqlite")
        task = AgentTask(task_id="t1", prompt="do something")
        output = AgentOutput(content="done!", agent="echo", task_id="t1", success=True)
        node = TaskNode(id="node-1", agent_class=EchoAgent, task=task)
        node.status = NodeStatus.DONE
        node.output = output

        store.save("run-abc", node)
        loaded = store.load_latest("run-abc")

        assert "node-1" in loaded
        status, loaded_output, error = loaded["node-1"]
        assert status == NodeStatus.DONE
        assert loaded_output is not None
        assert loaded_output.content == "done!"
        assert error is None

    def test_load_latest_returns_only_most_recent_per_node(self, tmp_path: Path) -> None:
        """If a node is checkpointed twice, load_latest returns only the latest row."""
        store = CheckpointStore(db_path=tmp_path / "cp.sqlite")
        task = AgentTask(task_id="t2", prompt="")

        # First checkpoint: FAILED
        node = TaskNode(id="flaky", agent_class=EchoAgent, task=task)
        node.status = NodeStatus.FAILED
        node.error = "first attempt failed"
        store.save("run-xyz", node)

        # Second checkpoint: DONE
        node.status = NodeStatus.DONE
        node.output = AgentOutput(content="recovered", agent="echo", task_id="t2")
        node.error = None
        store.save("run-xyz", node)

        loaded = store.load_latest("run-xyz")
        status, output, error = loaded["flaky"]
        assert status == NodeStatus.DONE
        assert output is not None
        assert output.content == "recovered"

    def test_load_latest_empty_for_unknown_run(self, tmp_path: Path) -> None:
        store = CheckpointStore(db_path=tmp_path / "cp.sqlite")
        loaded = store.load_latest("no-such-run")
        assert loaded == {}

    def test_save_failed_node_preserves_error(self, tmp_path: Path) -> None:
        store = CheckpointStore(db_path=tmp_path / "cp.sqlite")
        node = TaskNode(id="broken", agent_class=EchoAgent, task=AgentTask(prompt=""))
        node.status = NodeStatus.FAILED
        node.error = "boom"
        store.save("run-err", node)

        loaded = store.load_latest("run-err")
        _, _, error = loaded["broken"]
        assert error == "boom"


# ---------------------------------------------------------------------------
# Coordinator — checkpoint restore (rollback semantics)
# ---------------------------------------------------------------------------


class TestCoordinatorCheckpointRestore:
    @pytest.mark.asyncio
    async def test_completed_node_is_skipped_on_resume(
        self,
        registry: ToolRegistry,
        policy: TrustPolicy,
        memory: TieredMemory,
        tmp_path: Path,
    ) -> None:
        """A node that was DONE in a previous run must not be re-executed."""
        checkpoint_store = CheckpointStore(db_path=tmp_path / "cp.sqlite")
        run_id = "resume-run-1"

        # Track how many times node-a's agent is actually called
        call_counts: dict[str, int] = {"node-a": 0, "node-b": 0}

        class TrackedA(SubAgent):
            name = "tracked_a"
            description = ""

            async def run(self, task: AgentTask) -> AgentOutput:
                call_counts["node-a"] += 1
                return AgentOutput(content="fresh-a", agent=self.name, task_id=task.task_id)

        class TrackedB(SubAgent):
            name = "tracked_b"
            description = ""

            async def run(self, task: AgentTask) -> AgentOutput:
                call_counts["node-b"] += 1
                return AgentOutput(content="b-result", agent=self.name, task_id=task.task_id)

        # Pre-checkpoint node-a as DONE (simulating a prior run)
        prev_output = AgentOutput(
            content="from-checkpoint", agent="tracked_a", task_id="task-a", success=True
        )
        prev_node = TaskNode(id="node-a", agent_class=TrackedA, task=AgentTask(task_id="task-a", prompt="a"))
        prev_node.status = NodeStatus.DONE
        prev_node.output = prev_output
        checkpoint_store.save(run_id, prev_node)

        # Resume: build a fresh graph with the same run_id
        graph = TaskGraph(
            run_id=run_id,
            nodes=[
                TaskNode(id="node-a", agent_class=TrackedA, task=AgentTask(task_id="task-a", prompt="a")),
                TaskNode(
                    id="node-b",
                    agent_class=TrackedB,
                    task=AgentTask(task_id="task-b", prompt="b"),
                    depends_on=["node-a"],
                ),
            ],
        )

        coordinator = AgentCoordinator(
            registry=registry,
            policy=policy,
            memory=memory,
            checkpoint_store=checkpoint_store,
        )
        result = await coordinator.run(graph)

        # node-a was already DONE — must NOT have been re-executed
        assert call_counts["node-a"] == 0, "node-a should have been skipped (already checkpointed)"
        # node-b (depends on node-a) must have run
        assert call_counts["node-b"] == 1
        assert result.success
        # node-a's output in the final result comes from the checkpoint
        assert result.outputs["node-a"].content == "from-checkpoint"

    @pytest.mark.asyncio
    async def test_checkpoint_written_after_each_node_completes(
        self,
        registry: ToolRegistry,
        policy: TrustPolicy,
        memory: TieredMemory,
        tmp_path: Path,
    ) -> None:
        """After a successful run, each node must be checkpointed as DONE."""
        checkpoint_store = CheckpointStore(db_path=tmp_path / "cp.sqlite")

        graph = TaskGraph(
            nodes=[
                TaskNode(id="cp-a", agent_class=EchoAgent, task=AgentTask(prompt="alpha")),
                TaskNode(id="cp-b", agent_class=EchoAgent, task=AgentTask(prompt="beta")),
            ]
        )

        coordinator = AgentCoordinator(
            registry=registry,
            policy=policy,
            memory=memory,
            checkpoint_store=checkpoint_store,
        )
        result = await coordinator.run(graph)

        loaded = checkpoint_store.load_latest(graph.run_id)
        for node_id in ("cp-a", "cp-b"):
            assert node_id in loaded, f"Node {node_id!r} should be checkpointed"
            status, _, _ = loaded[node_id]
            assert status == NodeStatus.DONE


# ---------------------------------------------------------------------------
# Coordinator — subagent failure handling
# ---------------------------------------------------------------------------


class TestCoordinatorFailureHandling:
    @pytest.mark.asyncio
    async def test_failing_agent_output_marks_node_failed(
        self, coordinator: AgentCoordinator
    ) -> None:
        """An agent returning success=False should result in a FAILED node."""
        graph = TaskGraph(
            nodes=[
                TaskNode(
                    id="fail-node",
                    agent_class=FailingAgent,
                    task=AgentTask(prompt="doomed task"),
                    max_retries=0,
                )
            ]
        )
        result = await coordinator.run(graph)
        assert result.success is False
        assert "fail-node" in result.failed_nodes

    @pytest.mark.asyncio
    async def test_exploding_agent_does_not_crash_coordinator(
        self, coordinator: AgentCoordinator
    ) -> None:
        """An agent that raises an exception must be caught; coordinator returns gracefully."""
        graph = TaskGraph(
            nodes=[
                TaskNode(
                    id="bomb",
                    agent_class=ExplodingAgent,
                    task=AgentTask(prompt="will explode"),
                    max_retries=0,
                )
            ]
        )
        # Must not propagate the exception
        result = await coordinator.run(graph)
        assert isinstance(result, CoordinatorResult)
        assert result.success is False
        assert "bomb" in result.failed_nodes

    @pytest.mark.asyncio
    async def test_failed_node_does_not_block_independent_nodes(
        self,
        registry: ToolRegistry,
        policy: TrustPolicy,
        memory: TieredMemory,
        tmp_path: Path,
    ) -> None:
        """A FAILED node must not prevent independent (non-dependent) nodes from running."""
        checkpoint_store = CheckpointStore(db_path=tmp_path / "cp.sqlite")

        graph = TaskGraph(
            nodes=[
                TaskNode(
                    id="fail-independent",
                    agent_class=FailingAgent,
                    task=AgentTask(prompt="fail"),
                    max_retries=0,
                ),
                TaskNode(
                    id="succeed-independent",
                    agent_class=EchoAgent,
                    task=AgentTask(prompt="still runs"),
                ),
            ]
        )
        coordinator = AgentCoordinator(
            registry=registry,
            policy=policy,
            memory=memory,
            checkpoint_store=checkpoint_store,
        )
        result = await coordinator.run(graph)
        # The independent successful node should still have output
        assert "succeed-independent" in result.outputs
        assert result.outputs["succeed-independent"].content == "echo:still runs"

    @pytest.mark.asyncio
    async def test_coordinator_result_has_correct_run_id(
        self, coordinator: AgentCoordinator
    ) -> None:
        custom_run_id = "my-deterministic-run-id"
        graph = TaskGraph(
            run_id=custom_run_id,
            nodes=[TaskNode(id="n", agent_class=EchoAgent, task=AgentTask(prompt=""))],
        )
        result = await coordinator.run(graph)
        assert result.run_id == custom_run_id
